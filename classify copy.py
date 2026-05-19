#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# classify.py

import os
import gc
import json
import math
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# =========================================================
# CLUSTER / TF STABILITY
# =========================================================
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "8")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "8")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from Bio import SeqIO
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

# =========================================================
# Hardware / Mixed Precision Setup
# Keep same policy family as training so Transformer models load properly.
# =========================================================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    except Exception:
        pass

# =========================================================
# CONSTANTS
# =========================================================
DEFAULT_BASE_PATH = Path(os.environ.get("BASE_PATH", "/scratch/injamam/07-04-2026/Global"))
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
DEFAULT_MIN_PROB = 0.25
DEFAULT_MIN_PROB_MAP = json.dumps({
    "kingdom": 0.95,
    "phylum": 0.90,
    "class": 0.85,
    "order": 0.80,
    "family": 0.25,
    "genus": 0.25,
    "species": 0.25,
})
DEFAULT_FOLD = "all"
DEFAULT_METHOD = "CNN"
SUPPORTED_METHODS = ["CNN", "DBN"]

DEFAULT_FISH_KINGDOM = ["Metazoa"]
DEFAULT_FISH_PHYLUM = ["Chordata"]
DEFAULT_FISH_CLASSES = [
    "Actinopteri",
    "Chondrichthyes",
    "Myxini",
    "Cephalaspidomorphi",
    "Sarcopterygii",
    "Elasmobranchii",
    "Holostei",
]

# =========================================================
# REQUIRED: Custom Layer Definition for load_model()
# Keras 3-safe version with explicit build
# =========================================================
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.ff_dim = int(ff_dim)
        self.rate = float(rate)

        self.att = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim,
            name="mha",
        )
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(self.ff_dim, activation="relu", name="dense"),
                layers.Dense(self.embed_dim, name="dense_1"),
            ],
            name="ffn",
        )
        self.ln1 = layers.LayerNormalization(epsilon=1e-6, name="layer_normalization")
        self.ln2 = layers.LayerNormalization(epsilon=1e-6, name="layer_normalization_1")
        self.do1 = layers.Dropout(self.rate, name="dropout")
        self.do2 = layers.Dropout(self.rate, name="dropout_1")

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)

        # Build attention
        try:
            self.att.build(input_shape, input_shape, input_shape)
        except TypeError:
            try:
                self.att.build(input_shape, input_shape)
            except Exception:
                pass

        # Build FFN and norm layers explicitly so weights can load
        self.ffn.build(input_shape)
        self.ln1.build(input_shape)
        self.ln2.build(input_shape)

        super().build(input_shape)

    def call(self, inputs, training=False):
        input_dtype = inputs.dtype

        attn = self.att(inputs, inputs, training=training)
        attn = tf.cast(attn, input_dtype)
        attn = self.do1(attn, training=training)
        attn = tf.cast(attn, input_dtype)

        out1 = self.ln1(inputs + attn)
        out1 = tf.cast(out1, input_dtype)

        ffn = self.ffn(out1, training=training)
        ffn = tf.cast(ffn, input_dtype)
        ffn = self.do2(ffn, training=training)
        ffn = tf.cast(ffn, input_dtype)

        out2 = self.ln2(out1 + ffn)
        out2 = tf.cast(out2, input_dtype)

        return out2

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "rate": self.rate,
            }
        )
        return cfg

# =========================================================
# Taxonomy Tree
# =========================================================
@dataclass
class TaxonomyTree:
    ranks: List[str]
    mappings: Dict[str, Dict[str, int]]
    inv_mappings: Dict[str, Dict[int, str]]
    child_to_parent: Dict[str, Dict[int, int]]
    parent_to_children: Dict[str, Dict[int, List[int]]]

# =========================================================
# HELPERS
# =========================================================
def free_ram():
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()

def wrap_seq(seq: str, width: int = 80) -> str:
    s = str(seq).strip()
    return "\n".join(s[i:i + width] for i in range(0, len(s), width))

def fcgr_cpu(seq: str, k: int) -> np.ndarray:
    size = 1 << k
    mat = np.zeros((size, size), dtype=np.float32)
    seq = str(seq).upper().strip()
    n = len(seq)

    if n < k:
        return mat

    for i in range(n - k + 1):
        x = 0
        y = 0
        valid = True
        for j in range(k):
            mask = 1 << (k - 1 - j)
            c = seq[i + j]
            if c == "A":
                x |= mask
            elif c == "C":
                x |= mask
                y |= mask
            elif c == "G":
                y |= mask
            elif c == "T":
                pass
            else:
                valid = False
                break
        if valid:
            mat[y, x] += 1.0

    mx = float(np.max(mat))
    if mx > 0:
        mat /= mx
    return mat

def build_X(seqs: List[str], k: int) -> np.ndarray:
    h = 2 ** k
    X = np.empty((len(seqs), h, h, 1), dtype=np.float32)
    for i, s in enumerate(seqs):
        X[i, :, :, 0] = fcgr_cpu(s, k)
    return X

def load_tree(base_path: Path) -> TaxonomyTree:
    p = base_path / "parsed_files" / "taxonomy_tree.pkl"
    if not p.exists():
        raise FileNotFoundError(f"Missing taxonomy tree: {p}")
    with open(p, "rb") as f:
        return pickle.load(f)

def load_best_method(models_dir: Path, k: int) -> Optional[Tuple[str, int]]:
    p = models_dir / f"BEST_METHOD_k{k}.json"
    if not p.exists():
        return None
    try:
        with open(p, "r") as f:
            j = json.load(f)
        return str(j.get("method")), int(j.get("fold"))
    except Exception:
        return None

def list_available_folds(models_dir: Path, method: str, k: int) -> List[int]:
    folds = []
    for fp in sorted(models_dir.glob(f"{method}_k{k}_fold*.keras")):
        try:
            folds.append(int(fp.stem.split("_fold")[-1]))
        except Exception:
            pass
    return sorted(set(folds))

def safe_load_single_model(model_path: Path):
    print(f"Loading Model: {model_path}")
    custom_objects = {"TransformerBlock": TransformerBlock}
    try:
        return load_model(
            str(model_path),
            compile=False,
            safe_mode=False,
            custom_objects=custom_objects,
        )
    except TypeError:
        return load_model(
            str(model_path),
            compile=False,
            custom_objects=custom_objects,
        )

def load_models(models_dir: Path, method: str, k: int, fold_arg: str) -> Tuple[List[Any], List[int]]:
    available_folds = list_available_folds(models_dir, method, k)
    if not available_folds:
        raise FileNotFoundError(
            f"No model files found for method={method}, k={k} in {models_dir}"
        )

    if fold_arg == "all":
        selected_folds = available_folds
    elif fold_arg == "best":
        bm = load_best_method(models_dir, k)
        if bm and bm[0] == method and bm[1] in available_folds:
            selected_folds = [bm[1]]
        else:
            selected_folds = available_folds
    else:
        requested = int(fold_arg)
        if requested not in available_folds:
            raise FileNotFoundError(
                f"Requested fold {requested} not found for method={method}, k={k}. "
                f"Available folds: {available_folds}"
            )
        selected_folds = [requested]

    models = []
    for fold in selected_folds:
        model_path = models_dir / f"{method}_k{k}_fold{fold}.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")
        models.append(safe_load_single_model(model_path))

    return models, selected_folds

def average_outputs(output_list: List[Any]) -> List[np.ndarray]:
    if not output_list:
        raise ValueError("No outputs to average.")
    n_ranks = len(output_list[0])
    avg = []
    for r in range(n_ranks):
        stacked = np.stack([outs[r] for outs in output_list], axis=0)
        avg.append(np.mean(stacked, axis=0))
    return avg

def predict_ensemble(models: List[Any], X: np.ndarray, batch_size: int) -> List[np.ndarray]:
    outputs_all = []
    for i, model in enumerate(models, start=1):
        print(f"Running inference with model {i}/{len(models)} ...")
        outs = model.predict(X, batch_size=batch_size, verbose=1)
        outputs_all.append(outs)
    return average_outputs(outputs_all)

def parse_min_prob_map(raw: str, default_min_prob: float) -> Dict[str, float]:
    result = {r: float(default_min_prob) for r in RANKS}
    if str(raw).strip():
        mp = json.loads(str(raw))
        for k, v in (mp or {}).items():
            rk = str(k).strip().lower()
            if rk in result:
                result[rk] = float(v)
    return result

def names_to_indices(tree: TaxonomyTree, rank: str, names: List[str]) -> List[int]:
    result = []
    mapping = tree.mappings.get(rank, {})
    for n in names:
        if n in mapping:
            result.append(int(mapping[n]))
    return sorted(set(result))

def collect_descendants(tree: TaxonomyTree, start_rank: str, start_indices: List[int], target_rank: str) -> List[int]:
    if start_rank == target_rank:
        return sorted(set(start_indices))

    start_pos = RANKS.index(start_rank)
    target_pos = RANKS.index(target_rank)

    if target_pos < start_pos:
        return []

    current = set(start_indices)
    for pos in range(start_pos + 1, target_pos + 1):
        next_rank = RANKS[pos]
        new_current = set()
        for parent_idx in current:
            children = tree.parent_to_children.get(next_rank, {}).get(int(parent_idx), [])
            for c in children:
                new_current.add(int(c))
        current = new_current
        if not current:
            break

    return sorted(current)

def mask_renorm(probs: np.ndarray, allowed: Optional[List[int]]) -> np.ndarray:
    p = probs.astype(np.float32, copy=True)

    if allowed is None:
        return p

    if not allowed:
        p[:] = 0.0
        return p

    mask = np.ones_like(p, dtype=bool)
    idx = np.array(allowed, dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < p.shape[0])]
    mask[idx] = False
    p[mask] = 0.0

    s = float(p.sum())
    if s > 0:
        p /= s
    return p

def decode_with_constraints(
    probs_by_rank: Dict[str, np.ndarray],
    tree: TaxonomyTree,
    min_prob_by_rank: Dict[str, float],
    allowed_by_rank: Dict[str, Optional[List[int]]],
    non_fish_label: str = "Non-fish",
):
    labels: Dict[str, str] = {}
    scores: Dict[str, float] = {}
    indices: Dict[str, int] = {}

    prev = None

    for i, rank in enumerate(RANKS):
        probs = probs_by_rank[rank]

        # global mask
        probs = mask_renorm(probs, allowed_by_rank.get(rank))

        # hierarchical parent -> child mask
        if i > 0 and prev is not None:
            child_allowed = tree.parent_to_children.get(rank, {}).get(int(prev), [])
            probs = mask_renorm(probs, child_allowed)

        if probs.size == 0 or float(probs.sum()) <= 0.0:
            for j in range(i, len(RANKS)):
                rk = RANKS[j]
                labels[rk] = non_fish_label if rk in ("kingdom", "phylum", "class") else "Unclassified"
                scores[rk] = 0.0
                indices[rk] = -1
            break

        pi = int(np.argmax(probs))
        pv = float(probs[pi])

        if pv < float(min_prob_by_rank.get(rank, DEFAULT_MIN_PROB)):
            for j in range(i, len(RANKS)):
                rk = RANKS[j]
                labels[rk] = non_fish_label if rk in ("kingdom", "phylum", "class") else "Unclassified"
                scores[rk] = 0.0
                indices[rk] = -1
            break

        indices[rank] = pi
        labels[rank] = str(tree.inv_mappings[rank][pi])
        scores[rank] = pv
        prev = pi

    for rk in RANKS:
        if rk not in labels:
            labels[rk] = "Unclassified"
            scores[rk] = 0.0
            indices[rk] = -1

    return labels, scores, indices

# =========================================================
# MAIN
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--base-path", default=str(DEFAULT_BASE_PATH))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--method", default=DEFAULT_METHOD, help="CNN, DBN")
    ap.add_argument("--fold", default=str(DEFAULT_FOLD))
    ap.add_argument("--min-prob", type=float, default=DEFAULT_MIN_PROB)
    ap.add_argument("--min-prob-map", default=DEFAULT_MIN_PROB_MAP)
    ap.add_argument("--fish-only", action="store_true", help="Restrict decoding to fish-compatible taxonomy")
    ap.add_argument("--fish-kingdom", default=",".join(DEFAULT_FISH_KINGDOM))
    ap.add_argument("--fish-phylum", default=",".join(DEFAULT_FISH_PHYLUM))
    ap.add_argument("--fish-classes", default=",".join(DEFAULT_FISH_CLASSES))
    ap.add_argument("--out-prefix", default="classification_results")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    base = Path(args.base_path)
    models_dir = base / "models"
    tree = load_tree(base)

    method = str(args.method).strip()
    if method.lower() == "auto":
        bm = load_best_method(models_dir, int(args.k))
        method = bm[0] if bm else DEFAULT_METHOD

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"--method must be one of {SUPPORTED_METHODS} or auto")

    fold_arg = str(args.fold).strip().lower()
    models, used_folds = load_models(models_dir, method, int(args.k), fold_arg)
    print(f"Loaded method={method}, k={args.k}, folds={used_folds}")

    min_prob_by_rank = parse_min_prob_map(args.min_prob_map, args.min_prob)

    # Forward-only inference
    ids: List[str] = []
    seqs: List[str] = []
    with open(args.input, "r") as f:
        for rec in SeqIO.parse(f, "fasta"):
            ids.append(rec.id)
            seqs.append(str(rec.seq).upper().strip())

    if not seqs:
        raise ValueError("No sequences found in input FASTA.")

    if args.fish_only:
        fish_kingdom_names = [x.strip() for x in str(args.fish_kingdom).split(",") if x.strip()]
        fish_phylum_names = [x.strip() for x in str(args.fish_phylum).split(",") if x.strip()]
        fish_class_names = [x.strip() for x in str(args.fish_classes).split(",") if x.strip()]

        fish_kingdom_idx = names_to_indices(tree, "kingdom", fish_kingdom_names)
        fish_phylum_idx = names_to_indices(tree, "phylum", fish_phylum_names)
        fish_class_idx = names_to_indices(tree, "class", fish_class_names)

        fish_order_idx = collect_descendants(tree, "class", fish_class_idx, "order")
        fish_family_idx = collect_descendants(tree, "class", fish_class_idx, "family")
        fish_genus_idx = collect_descendants(tree, "class", fish_class_idx, "genus")
        fish_species_idx = collect_descendants(tree, "class", fish_class_idx, "species")

        allowed_by_rank: Dict[str, Optional[List[int]]] = {
            "kingdom": fish_kingdom_idx if fish_kingdom_idx else None,
            "phylum": fish_phylum_idx if fish_phylum_idx else None,
            "class": fish_class_idx if fish_class_idx else None,
            "order": fish_order_idx if fish_order_idx else None,
            "family": fish_family_idx if fish_family_idx else None,
            "genus": fish_genus_idx if fish_genus_idx else None,
            "species": fish_species_idx if fish_species_idx else None,
        }
    else:
        allowed_by_rank = {r: None for r in RANKS}

    print("Generating FCGR tensors (Forward only)...")
    X = build_X(seqs, int(args.k))

    print("Running ensemble inference (Forward only)...")
    outs = predict_ensemble(models, X, args.batch_size)

    labels_all = []
    scores_all = []

    print("Decoding taxonomy...")
    for i in range(len(seqs)):
        probs = {RANKS[j]: outs[j][i] for j in range(7)}

        if args.fish_only:
            lbl, sc, _ = decode_with_constraints(
                probs_by_rank=probs,
                tree=tree,
                min_prob_by_rank=min_prob_by_rank,
                allowed_by_rank=allowed_by_rank,
                non_fish_label="Non-fish",
            )
        else:
            lbl, sc, _ = decode_with_constraints(
                probs_by_rank=probs,
                tree=tree,
                min_prob_by_rank=min_prob_by_rank,
                allowed_by_rank=allowed_by_rank,
                non_fish_label="Unclassified",
            )

        labels_all.append(lbl)
        scores_all.append(sc)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    out_txt = out_prefix.with_suffix(".txt")
    out_csv = out_prefix.with_suffix(".csv")

    with open(out_txt, "w") as f:
        for i, sid in enumerate(ids):
            parts = [f">{sid} 1", "ORIENT FWD"]
            for r in RANKS:
                parts.append(f"{labels_all[i][r]} {scores_all[i][r]:.4f}")
            f.write("|".join(parts) + "\n")
            f.write(wrap_seq(seqs[i], 80) + "\n")

    data = {
        "seq_id": ids,
        "orientation": ["FWD"] * len(ids),
    }
    for r in RANKS:
        data[f"{r}_label"] = [d[r] for d in labels_all]
        data[f"{r}_prob"] = [float(d[r]) for d in scores_all]

    pd.DataFrame(data).to_csv(out_csv, index=False)

    print("? Success! Classification complete.")
    print(f"Wrote TXT: {out_txt}")
    print(f"Wrote CSV: {out_csv}")

    free_ram()

if __name__ == "__main__":
    main()