#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FISHNET-META classifier

Public, portable classifier script for:
FISHNET-META: Fish taxonomic Identification using sequence-based hierarchical
neural network for 12S eDNA METAbarcoding.

Default behavior:
- Uses the official 10-fold CNN ensemble.
- User selects only the model region: global or atlantic.
- Model files are loaded from ~/.fishnet-meta/models/<region>/ by default.
- No hardcoded server path is required.

Expected model package layout:

~/.fishnet-meta/models/
├── global/
│   ├── models/
│   │   ├── CNN_k8_fold0.keras
│   │   ├── ...
│   │   └── CNN_k8_fold9.keras
│   └── parsed_files/
│       └── taxonomy_tree.pkl
└── atlantic/
    ├── models/
    │   ├── CNN_k8_fold0.keras
    │   ├── ...
    │   └── CNN_k8_fold9.keras
    └── parsed_files/
        └── taxonomy_tree.pkl
"""

import os
import gc
import sys
import json
import tarfile
import pickle
import shutil
import argparse
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# =========================================================
# TensorFlow stability defaults
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
# Constants
# =========================================================
APP_NAME = "FISHNET-META"
APP_VERSION = "1.0.0"

RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

DEFAULT_MODEL_HOME = Path(
    os.environ.get("FISHNET_META_MODEL_HOME", Path.home() / ".fishnet-meta" / "models")
)

DEFAULT_REGION = "global"
DEFAULT_K = 8
DEFAULT_METHOD = "CNN"
DEFAULT_FOLD_MODE = "all"
SUPPORTED_METHODS = ["CNN"]

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

# Replace these URLs after uploading your model archives to Zenodo.
# Zenodo direct file URL format:
# https://zenodo.org/records/<RECORD_ID>/files/<FILE_NAME>?download=1
MODEL_REGISTRY = {
    "global": {
        "version": "1.0.0",
        "archive_name": "fishnet-meta-global-v1.0.0.tar.gz",
        "url": "https://zenodo.org/records/REPLACE_GLOBAL_RECORD_ID/files/fishnet-meta-global-v1.0.0.tar.gz?download=1",
    },
    "atlantic": {
        "version": "1.0.0",
        "archive_name": "fishnet-meta-atlantic-v1.0.0.tar.gz",
        "url": "https://zenodo.org/records/REPLACE_ATLANTIC_RECORD_ID/files/fishnet-meta-atlantic-v1.0.0.tar.gz?download=1",
    },
}


# =========================================================
# Console helpers
# =========================================================
def info(message: str) -> None:
    print(f"[INFO] {message}")


def warn(message: str) -> None:
    print(f"[WARNING] {message}")


def fail(message: str, exit_code: int = 1) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
    raise SystemExit(exit_code)


# =========================================================
# Hardware / mixed precision setup
# =========================================================
def configure_tensorflow() -> None:
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        info(f"Detected {len(gpus)} GPU device(s). Enabling memory growth.")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as exc:
                warn(f"Could not set GPU memory growth: {exc}")

        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            info("Using mixed_float16 policy for GPU inference.")
        except Exception as exc:
            warn(f"Could not set mixed precision policy: {exc}")
    else:
        warn("No GPU detected. FISHNET-META will run on CPU. This may be slow for the full 10-fold ensemble.")


# =========================================================
# Custom layer required for loading some trained Keras models
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

        try:
            self.att.build(input_shape, input_shape, input_shape)
        except TypeError:
            try:
                self.att.build(input_shape, input_shape)
            except Exception:
                pass

        self.ffn.build(input_shape)
        self.ln1.build(input_shape)
        self.ln2.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=False):
        input_dtype = inputs.dtype

        attn = self.att(inputs, inputs, training=training)
        attn = tf.cast(attn, input_dtype)
        attn = self.do1(attn, training=training)

        out1 = self.ln1(inputs + attn)
        out1 = tf.cast(out1, input_dtype)

        ffn = self.ffn(out1, training=training)
        ffn = tf.cast(ffn, input_dtype)
        ffn = self.do2(ffn, training=training)

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
# Taxonomy tree object
# =========================================================
@dataclass
class TaxonomyTree:
    ranks: List[str]
    mappings: Dict[str, Dict[str, int]]
    inv_mappings: Dict[str, Dict[int, str]]
    child_to_parent: Dict[str, Dict[int, int]]
    parent_to_children: Dict[str, Dict[int, List[int]]]


# =========================================================
# Model download helpers
# =========================================================
def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    info(f"Downloading:")
    info(f"  {url}")
    info(f"Saving to:")
    info(f"  {destination}")

    try:
        with urllib.request.urlopen(url) as response:
            total = response.length
            downloaded = 0
            block_size = 1024 * 1024

            with open(destination, "wb") as out:
                while True:
                    block = response.read(block_size)
                    if not block:
                        break
                    out.write(block)
                    downloaded += len(block)

                    if total:
                        percent = downloaded * 100 / total
                        print(f"\rDownloaded {downloaded / (1024 ** 3):.2f} GB / {total / (1024 ** 3):.2f} GB ({percent:.1f}%)", end="")
                    else:
                        print(f"\rDownloaded {downloaded / (1024 ** 3):.2f} GB", end="")

            print()

    except Exception as exc:
        if destination.exists():
            destination.unlink()
        raise RuntimeError(f"Download failed: {exc}") from exc


def safe_extract_tar_gz(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_root = target_dir.resolve()

    info(f"Extracting {archive_path} to {target_dir}")

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (target_dir / member.name).resolve()
            if not str(member_path).startswith(str(target_root)):
                raise RuntimeError(f"Unsafe path detected inside archive: {member.name}")
        tar.extractall(target_dir)

    info("Extraction complete.")


def download_models(region: str, model_home: Path, force: bool = False) -> None:
    region = region.lower().strip()

    if region not in MODEL_REGISTRY:
        fail(f"Unknown region '{region}'. Choose one of: {', '.join(MODEL_REGISTRY.keys())}")

    item = MODEL_REGISTRY[region]
    region_dir = model_home / region

    if is_model_package_ready(region_dir):
        if not force:
            info(f"Model package already exists and looks ready: {region_dir}")
            info("Use --force to re-download.")
            return

        warn(f"Removing existing model package because --force was used: {region_dir}")
        shutil.rmtree(region_dir)

    url = item["url"]

    if "REPLACE_" in url:
        fail(
            "The model download URL has not been configured yet.\n"
            "Upload the model archive to Zenodo first, then replace the URL in MODEL_REGISTRY.\n"
            "Expected Zenodo URL format:\n"
            "https://zenodo.org/records/<RECORD_ID>/files/<FILE_NAME>?download=1"
        )

    archive_path = model_home / item["archive_name"]
    download_file(url, archive_path)
    safe_extract_tar_gz(archive_path, model_home)

    if not is_model_package_ready(region_dir):
        fail(
            f"Downloaded archive was extracted, but the model package is not in the expected layout.\n"
            f"Expected:\n"
            f"  {region_dir / 'models'}\n"
            f"  {region_dir / 'parsed_files' / 'taxonomy_tree.pkl'}"
        )

    info(f"Model package is ready: {region_dir}")


def is_model_package_ready(region_dir: Path) -> bool:
    models_dir = region_dir / "models"
    parsed_dir = region_dir / "parsed_files"
    taxonomy_tree = parsed_dir / "taxonomy_tree.pkl"

    if not models_dir.exists():
        return False

    if not taxonomy_tree.exists():
        return False

    model_files = sorted(models_dir.glob("CNN_k8_fold*.keras"))
    return len(model_files) >= 10


# =========================================================
# Core helpers
# =========================================================
def free_ram() -> None:
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


def load_tree(region_base: Path) -> TaxonomyTree:
    p = region_base / "parsed_files" / "taxonomy_tree.pkl"

    if not p.exists():
        raise FileNotFoundError(
            f"Missing taxonomy tree:\n"
            f"  {p}\n\n"
            f"Expected model package layout:\n"
            f"  {region_base / 'models'}\n"
            f"  {region_base / 'parsed_files' / 'taxonomy_tree.pkl'}"
        )

    with open(p, "rb") as f:
        return pickle.load(f)


def list_available_folds(models_dir: Path, method: str, k: int) -> List[int]:
    folds = []

    for fp in sorted(models_dir.glob(f"{method}_k{k}_fold*.keras")):
        try:
            folds.append(int(fp.stem.split("_fold")[-1]))
        except Exception:
            pass

    return sorted(set(folds))


def safe_load_single_model(model_path: Path):
    info(f"Loading model: {model_path.name}")

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


def load_models(models_dir: Path, method: str, k: int, developer_fold: Optional[str] = None) -> Tuple[List[Any], List[int]]:
    available_folds = list_available_folds(models_dir, method, k)

    if not available_folds:
        raise FileNotFoundError(
            f"No model files found.\n"
            f"Folder checked: {models_dir}\n"
            f"Expected files like: {method}_k{k}_fold0.keras ... {method}_k{k}_fold9.keras"
        )

    if developer_fold:
        developer_fold = developer_fold.strip().lower()
        if developer_fold == "all":
            selected_folds = available_folds
        else:
            requested = int(developer_fold)
            if requested not in available_folds:
                raise FileNotFoundError(
                    f"Requested fold {requested} not found. Available folds: {available_folds}"
                )
            selected_folds = [requested]
    else:
        selected_folds = available_folds

    if len(selected_folds) < 10:
        warn(
            f"Only {len(selected_folds)} fold model(s) will be used: {selected_folds}. "
            f"The official FISHNET-META mode uses all 10 folds."
        )

    models = []

    for fold in selected_folds:
        model_path = models_dir / f"{method}_k{k}_fold{fold}.keras"

        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")

        models.append(safe_load_single_model(model_path))

    return models, selected_folds


def average_outputs(output_list: List[Any]) -> List[np.ndarray]:
    if not output_list:
        raise ValueError("No model outputs to average.")

    n_ranks = len(output_list[0])
    avg = []

    for r in range(n_ranks):
        stacked = np.stack([outs[r] for outs in output_list], axis=0)
        avg.append(np.mean(stacked, axis=0))

    return avg


def predict_ensemble(models: List[Any], X: np.ndarray, batch_size: int) -> List[np.ndarray]:
    outputs_all = []

    for i, model in enumerate(models, start=1):
        info(f"Running inference with fold model {i}/{len(models)} ...")
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

        probs = mask_renorm(probs, allowed_by_rank.get(rank))

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


def read_fasta(input_path: Path) -> Tuple[List[str], List[str]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input FASTA file not found: {input_path}")

    ids: List[str] = []
    seqs: List[str] = []

    with open(input_path, "r", encoding="utf-8") as f:
        for rec in SeqIO.parse(f, "fasta"):
            ids.append(rec.id)
            seqs.append(str(rec.seq).upper().strip())

    if not seqs:
        raise ValueError(
            f"No sequences were found in the input FASTA file:\n"
            f"  {input_path}\n\n"
            f"Example FASTA format:\n"
            f">sample_001\n"
            f"ACTGACTGACTGACTG"
        )

    return ids, seqs


def resolve_region_base(args) -> Path:
    region = str(args.region).strip().lower()

    if args.base_path:
        return Path(args.base_path).expanduser().resolve()

    return Path(args.model_home).expanduser().resolve() / region


def validate_region_package(region_base: Path, region: str) -> None:
    if not region_base.exists():
        raise FileNotFoundError(
            f"Model package for region '{region}' was not found.\n\n"
            f"Expected folder:\n"
            f"  {region_base}\n\n"
            f"Download it first:\n"
            f"  python classify.py download-models --region {region}\n\n"
            f"Or provide a direct folder:\n"
            f"  python classify.py classify --input input.fasta --region {region} --base-path /path/to/{region}"
        )

    models_dir = region_base / "models"
    parsed_dir = region_base / "parsed_files"
    taxonomy_tree = parsed_dir / "taxonomy_tree.pkl"

    if not models_dir.exists():
        raise FileNotFoundError(f"Missing models folder: {models_dir}")

    if not taxonomy_tree.exists():
        raise FileNotFoundError(f"Missing taxonomy tree file: {taxonomy_tree}")


def classify(args) -> None:
    configure_tensorflow()

    region = str(args.region).strip().lower()
    region_base = resolve_region_base(args)
    validate_region_package(region_base, region)

    models_dir = region_base / "models"

    info(f"{APP_NAME} v{APP_VERSION}")
    info(f"Region: {region}")
    info(f"Model package: {region_base}")
    info("Mode: official 10-fold ensemble")

    method = str(args.method).strip().upper()

    if method not in SUPPORTED_METHODS:
        raise ValueError(f"--method must be one of {SUPPORTED_METHODS}")

    tree = load_tree(region_base)

    models, used_folds = load_models(
        models_dir=models_dir,
        method=method,
        k=int(args.k),
        developer_fold=args.developer_fold,
    )

    info(f"Loaded method={method}, k={args.k}, folds={used_folds}")

    min_prob_by_rank = parse_min_prob_map(args.min_prob_map, args.min_prob)

    input_path = Path(args.input).expanduser().resolve()
    ids, seqs = read_fasta(input_path)

    info(f"Input sequences: {len(seqs)}")
    info("Generating FCGR tensors...")
    X = build_X(seqs, int(args.k))

    info("Running ensemble inference...")
    outs = predict_ensemble(models, X, args.batch_size)

    if args.fish_only:
        fish_kingdom_names = [x.strip() for x in str(args.fish_kingdom).split(",") if x.strip()]
        fish_phylum_names = [x.strip() for x in str(args.fish_phylum).split(",") if x.strip()]
        fish_class_names = [x.strip() for x in str(args.fish_classes).split(",") if x.strip()]

        fish_kingdom_idx = names_to_indices(tree, "kingdom", fish_kingdom_names)
        fish_phylum_idx = names_to_indices(tree, "phylum", fish_phylum_names)
        fish_class_idx = names_to_indices(tree, "class", fish_class_names)

        allowed_by_rank: Dict[str, Optional[List[int]]] = {
            "kingdom": fish_kingdom_idx if fish_kingdom_idx else None,
            "phylum": fish_phylum_idx if fish_phylum_idx else None,
            "class": fish_class_idx if fish_class_idx else None,
            "order": collect_descendants(tree, "class", fish_class_idx, "order") if fish_class_idx else None,
            "family": collect_descendants(tree, "class", fish_class_idx, "family") if fish_class_idx else None,
            "genus": collect_descendants(tree, "class", fish_class_idx, "genus") if fish_class_idx else None,
            "species": collect_descendants(tree, "class", fish_class_idx, "species") if fish_class_idx else None,
        }
    else:
        allowed_by_rank = {r: None for r in RANKS}

    labels_all = []
    scores_all = []

    info("Decoding taxonomy...")

    for i in range(len(seqs)):
        probs = {RANKS[j]: outs[j][i] for j in range(7)}

        lbl, sc, _ = decode_with_constraints(
            probs_by_rank=probs,
            tree=tree,
            min_prob_by_rank=min_prob_by_rank,
            allowed_by_rank=allowed_by_rank,
            non_fish_label="Non-fish" if args.fish_only else "Unclassified",
        )

        labels_all.append(lbl)
        scores_all.append(sc)

    out_prefix = Path(args.out_prefix).expanduser()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    out_txt = out_prefix.with_suffix(".txt")
    out_csv = out_prefix.with_suffix(".csv")
    out_json = out_prefix.with_suffix(".json")
    out_summary = out_prefix.parent / f"{out_prefix.name}_summary.txt"

    with open(out_txt, "w", encoding="utf-8") as f:
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

    df = pd.DataFrame(data)
    df.to_csv(out_csv, index=False)

    records = df.to_dict(orient="records")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    classified_species = sum(1 for d in labels_all if d["species"] not in ("Unclassified", "Non-fish"))
    classified_genus = sum(1 for d in labels_all if d["genus"] not in ("Unclassified", "Non-fish"))

    with open(out_summary, "w", encoding="utf-8") as f:
        f.write(f"{APP_NAME} classification summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Input file: {input_path}\n")
        f.write(f"Region: {region}\n")
        f.write(f"Method: {method}\n")
        f.write(f"k: {args.k}\n")
        f.write(f"Folds used: {used_folds}\n")
        f.write(f"Total sequences: {len(ids)}\n")
        f.write(f"Genus-level classified: {classified_genus}\n")
        f.write(f"Species-level classified: {classified_species}\n")

    info("Classification complete.")
    info(f"Wrote TXT: {out_txt}")
    info(f"Wrote CSV: {out_csv}")
    info(f"Wrote JSON: {out_json}")
    info(f"Wrote summary: {out_summary}")

    free_ram()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fishnet-meta",
        description="FISHNET-META: 12S fish eDNA taxonomic classifier using a 10-fold deep-learning ensemble.",
    )

    subparsers = parser.add_subparsers(dest="command")

    classify_parser = subparsers.add_parser(
        "classify",
        help="Classify sequences from an input FASTA file.",
    )
    classify_parser.add_argument("--input", required=True, help="Input FASTA file.")
    classify_parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        choices=["global", "atlantic"],
        help="Model region to use. Default: global.",
    )
    classify_parser.add_argument(
        "--model-home",
        default=str(DEFAULT_MODEL_HOME),
        help="Folder containing downloaded model packages. Default: ~/.fishnet-meta/models",
    )
    classify_parser.add_argument(
        "--base-path",
        default=None,
        help="Advanced: direct path to one region package folder.",
    )
    classify_parser.add_argument("--k", type=int, default=DEFAULT_K, help="k-mer size. Default: 8.")
    classify_parser.add_argument("--method", default=DEFAULT_METHOD, help="Model method. Default: CNN.")
    classify_parser.add_argument("--min-prob", type=float, default=DEFAULT_MIN_PROB)
    classify_parser.add_argument("--min-prob-map", default=DEFAULT_MIN_PROB_MAP)
    classify_parser.add_argument("--fish-only", action="store_true", help="Restrict decoding to fish-compatible taxonomy.")
    classify_parser.add_argument("--fish-kingdom", default=",".join(DEFAULT_FISH_KINGDOM))
    classify_parser.add_argument("--fish-phylum", default=",".join(DEFAULT_FISH_PHYLUM))
    classify_parser.add_argument("--fish-classes", default=",".join(DEFAULT_FISH_CLASSES))
    classify_parser.add_argument("--out-prefix", default="results/classification_results")
    classify_parser.add_argument("--batch-size", type=int, default=32)
    classify_parser.add_argument(
        "--developer-fold",
        default=None,
        help=argparse.SUPPRESS,
    )

    download_parser = subparsers.add_parser(
        "download-models",
        help="Download FISHNET-META model files.",
    )
    download_parser.add_argument(
        "--region",
        required=True,
        choices=["global", "atlantic"],
        help="Model region to download.",
    )
    download_parser.add_argument(
        "--model-home",
        default=str(DEFAULT_MODEL_HOME),
        help="Where to save model packages. Default: ~/.fishnet-meta/models",
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the model package already exists.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "download-models":
        download_models(
            region=args.region,
            model_home=Path(args.model_home).expanduser().resolve(),
            force=args.force,
        )
    elif args.command == "classify":
        classify(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
