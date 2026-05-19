# FISHNET-META Model Structure

The public code expects this model folder layout:

```text
~/.fishnet-meta/models/<region>/
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

Supported regions:

```text
global
atlantic
```

Default model home:

```text
~/.fishnet-meta/models/
```

You can override it:

```bash
python fishnet_meta/classify.py classify \
  --input examples/input_template.fasta \
  --region global \
  --model-home /path/to/models \
  --out-prefix results/example
```

Or provide a direct path to one region:

```bash
python fishnet_meta/classify.py classify \
  --input examples/input_template.fasta \
  --region global \
  --base-path /path/to/global \
  --out-prefix results/example
```
