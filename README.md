# FISHNET-META

**Fish taxonomic Identification using sequence-based hierarchical neural network for 12S eDNA METAbarcoding**

Developed and maintained by **BioMac Lab**  
Website: <https://www.biomaclab.com>

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange)
![Model](https://img.shields.io/badge/model-10--fold%20CNN%20ensemble-success)
![Domain](https://img.shields.io/badge/domain-12S%20eDNA%20metabarcoding-green)
![Maintained by BioMac Lab](https://img.shields.io/badge/maintained%20by-BioMac%20Lab-blue)

---

## Overview

**FISHNET-META** is a deep-learning classifier for fish taxonomic identification from **12S eDNA metabarcoding sequences**.

It is designed as a practical classifier for researchers who want to classify unknown fish 12S sequences from FASTA files. The tool uses trained CNN fold models and a hierarchical taxonomy decoder to predict taxonomy from higher ranks down to species level.

FISHNET-META is intended to be used like other taxonomic classifiers: the user provides an input FASTA file, chooses a trained model region, and receives classification output files.

---

## Main features

- Classifies fish 12S eDNA sequences from FASTA files
- Supports Global and Atlantic trained model packages
- Uses the official 10-fold CNN ensemble by default
- No need for users to choose a fold manually
- Produces TXT, CSV, JSON, and summary outputs
- Provides rank-wise labels and probabilities
- Supports optional fish-only hierarchical decoding
- Works on Linux, macOS, and Windows
- Designed for GitHub + Zenodo public release

---

## Model packages

FISHNET-META currently supports two model regions.

| Region | Use case |
|---|---|
| `global` | Broad fish classification using the Global reference/training space |
| `atlantic` | Atlantic-focused fish classification |

The model files are large, so they are **not stored inside this GitHub repository**.

Default model location after download:

```text
~/.fishnet-meta/models/
```

Expected folder structure:

```text
~/.fishnet-meta/models/
├── global/
│   ├── models/
│   │   ├── CNN_k8_fold0.keras
│   │   ├── CNN_k8_fold1.keras
│   │   ├── CNN_k8_fold2.keras
│   │   ├── CNN_k8_fold3.keras
│   │   ├── CNN_k8_fold4.keras
│   │   ├── CNN_k8_fold5.keras
│   │   ├── CNN_k8_fold6.keras
│   │   ├── CNN_k8_fold7.keras
│   │   ├── CNN_k8_fold8.keras
│   │   └── CNN_k8_fold9.keras
│   └── parsed_files/
│       └── taxonomy_tree.pkl
└── atlantic/
    ├── models/
    │   ├── CNN_k8_fold0.keras
    │   ├── CNN_k8_fold1.keras
    │   ├── CNN_k8_fold2.keras
    │   ├── CNN_k8_fold3.keras
    │   ├── CNN_k8_fold4.keras
    │   ├── CNN_k8_fold5.keras
    │   ├── CNN_k8_fold6.keras
    │   ├── CNN_k8_fold7.keras
    │   ├── CNN_k8_fold8.keras
    │   └── CNN_k8_fold9.keras
    └── parsed_files/
        └── taxonomy_tree.pkl
```

---

## Why all 10 folds are used

FISHNET-META uses all 10 fold models as the official classifier ensemble.

This means users do **not** need to select a fold. The software loads all available fold models, runs inference with each model, averages the prediction probabilities, and then decodes the final hierarchical taxonomy.

This is the recommended public mode because it is more stable than using only one fold model.

---

## Installation

### Option 1: Conda installation

```bash
git clone https://github.com/BioMacLab/fishnet-meta.git
cd fishnet-meta

conda env create -f environment.yml
conda activate fishnet-meta
```

### Option 2: pip installation

```bash
git clone https://github.com/BioMacLab/fishnet-meta.git
cd fishnet-meta

python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

## Download model files

After uploading the model archives to Zenodo, update the download URLs inside `fishnet_meta/classify.py`.

Then users can download models like this:

### Download Global model

```bash
python fishnet_meta/classify.py download-models --region global
```

### Download Atlantic model

```bash
python fishnet_meta/classify.py download-models --region atlantic
```

---

## Input FASTA format

Yes, you should include an input FASTA template for users.

Example:

```fasta
>sample_001
ACTGACTGACTGACTGACTGACTGACTGACTGACTGACTG
>sample_002
TTGACCTGACTGACTGACTGACTGACTGACTGACTGACTG
```

Rules:

- Use FASTA format.
- Each sequence must start with `>sequence_id`.
- Use DNA letters: `A`, `C`, `G`, `T`.
- Ambiguous letters may be ignored during FCGR generation.
- Very short sequences may not classify well.

A small example file is included here:

```text
examples/input_template.fasta
```

---

## Run classification

### Global model

```bash
python fishnet_meta/classify.py classify \
  --input examples/input_template.fasta \
  --region global \
  --out-prefix results/example_global
```

### Atlantic model

```bash
python fishnet_meta/classify.py classify \
  --input examples/input_template.fasta \
  --region atlantic \
  --out-prefix results/example_atlantic
```

### With fish-only decoding

```bash
python fishnet_meta/classify.py classify \
  --input examples/input_template.fasta \
  --region global \
  --fish-only \
  --out-prefix results/example_global_fish_only
```

---

## Output files

For this command:

```bash
python fishnet_meta/classify.py classify \
  --input examples/input_template.fasta \
  --region global \
  --out-prefix results/example_global
```

FISHNET-META creates:

```text
results/example_global.txt
results/example_global.csv
results/example_global.json
results/example_global_summary.txt
```

### CSV columns

```text
seq_id
orientation
kingdom_label
kingdom_prob
phylum_label
phylum_prob
class_label
class_prob
order_label
order_prob
family_label
family_prob
genus_label
genus_prob
species_label
species_prob
```

---

## Example output

Example CSV-like result:

```text
seq_id,orientation,kingdom_label,kingdom_prob,phylum_label,phylum_prob,class_label,class_prob,order_label,order_prob,family_label,family_prob,genus_label,genus_prob,species_label,species_prob
sample_001,FWD,Metazoa,0.9999,Chordata,0.9987,Actinopteri,0.9912,Unclassified,0.0000,Unclassified,0.0000,Unclassified,0.0000,Unclassified,0.0000
```

---

## Zenodo model hosting plan

Do not upload model files directly to GitHub. The `.keras` model files are too large for normal GitHub storage.

Recommended Zenodo archive names:

```text
fishnet-meta-global-v1.0.0.tar.gz
fishnet-meta-atlantic-v1.0.0.tar.gz
```

Recommended Zenodo direct download URL format:

```text
https://zenodo.org/records/<RECORD_ID>/files/fishnet-meta-global-v1.0.0.tar.gz?download=1
https://zenodo.org/records/<RECORD_ID>/files/fishnet-meta-atlantic-v1.0.0.tar.gz?download=1
```

After Zenodo upload, replace:

```text
REPLACE_GLOBAL_RECORD_ID
REPLACE_ATLANTIC_RECORD_ID
```

inside `fishnet_meta/classify.py`.

---

## Recommended repository structure

```text
fishnet-meta/
├── README.md
├── LICENSE
├── MODEL_LICENSE.md
├── CITATION.cff
├── environment.yml
├── requirements.txt
├── .gitignore
├── fishnet_meta/
│   ├── __init__.py
│   └── classify.py
├── examples/
│   └── input_template.fasta
└── docs/
    ├── zenodo_upload_guide.md
    └── model_structure.md
```

---

## Troubleshooting

### Error: model package was not found

Run:

```bash
python fishnet_meta/classify.py download-models --region global
```

or:

```bash
python fishnet_meta/classify.py download-models --region atlantic
```

### Error: Zenodo URL has not been configured yet

This means the Zenodo record ID is still a placeholder. Upload the model archive to Zenodo first and replace the URL in `MODEL_REGISTRY`.

### Error: no GPU detected

The classifier can run on CPU, but the full 10-fold ensemble may be slow. A GPU workstation, cloud GPU, or HPC server is recommended.

### Error: no sequences found

Check that your input file is valid FASTA format:

```fasta
>sample_001
ACTGACTGACTGACTG
```

### Out of memory

The official classifier uses all 10 fold models. For large input files, try a smaller batch size:

```bash
python fishnet_meta/classify.py classify \
  --input samples.fasta \
  --region global \
  --batch-size 4 \
  --out-prefix results/samples
```

---

## Citation

If you use FISHNET-META, please cite:

```text
FISHNET-META: Fish taxonomic Identification using sequence-based hierarchical neural network for 12S eDNA METAbarcoding.
BioMac Lab. https://www.biomaclab.com
```

After the manuscript or Zenodo DOI is available, update this section with the final DOI.

---

## License

Source code license: see `LICENSE`.

Model weights license: see `MODEL_LICENSE.md`.

The BioMac Lab name, BioMac Lab logo, and FISHNET-META name may not be used to suggest endorsement or official partnership without written permission from BioMac Lab.

---

## Contact

BioMac Lab  
Website: <https://www.biomaclab.com>
