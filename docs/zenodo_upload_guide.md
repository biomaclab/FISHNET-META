# Zenodo Upload Guide for FISHNET-META Models

## 1. Prepare model folders

Create this layout before compression:

```text
global/
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

For Atlantic:

```text
atlantic/
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

Rename your Atlantic taxonomy file from:

```text
taxonomy_tree_atlantic.pkl
```

to:

```text
taxonomy_tree.pkl
```

inside:

```text
atlantic/parsed_files/
```

## 2. Compress archives

Linux/macOS/Git Bash:

```bash
tar -czf fishnet-meta-global-v1.0.0.tar.gz global
tar -czf fishnet-meta-atlantic-v1.0.0.tar.gz atlantic
```

Windows PowerShell with 7-Zip installed:

```powershell
7z a -ttar fishnet-meta-global-v1.0.0.tar global
7z a -tgzip fishnet-meta-global-v1.0.0.tar.gz fishnet-meta-global-v1.0.0.tar

7z a -ttar fishnet-meta-atlantic-v1.0.0.tar atlantic
7z a -tgzip fishnet-meta-atlantic-v1.0.0.tar.gz fishnet-meta-atlantic-v1.0.0.tar
```

## 3. Upload to Zenodo

Recommended title:

```text
FISHNET-META v1.0.0 Trained Model Weights for 12S Fish eDNA Classification
```

Recommended upload type:

```text
Dataset
```

Recommended files:

```text
fishnet-meta-global-v1.0.0.tar.gz
fishnet-meta-atlantic-v1.0.0.tar.gz
```

Recommended description:

```text
This Zenodo record contains the trained model packages for FISHNET-META, a
deep-learning based 12S fish eDNA metabarcoding classifier developed by
BioMac Lab. The release includes Global and Atlantic 10-fold CNN ensemble
model packages and required taxonomy tree files for public inference.
```

Recommended keywords:

```text
FISHNET-META, BioMac Lab, fish taxonomy, 12S, eDNA, metabarcoding, deep learning, CNN, bioinformatics
```

## 4. Direct download URL format

After publishing, Zenodo will give a record ID.

Use this format:

```text
https://zenodo.org/records/<RECORD_ID>/files/fishnet-meta-global-v1.0.0.tar.gz?download=1
https://zenodo.org/records/<RECORD_ID>/files/fishnet-meta-atlantic-v1.0.0.tar.gz?download=1
```

Then replace the placeholder URLs in:

```text
fishnet_meta/classify.py
```

inside `MODEL_REGISTRY`.
