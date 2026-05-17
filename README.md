# AquaTaxa: Deep-Learning Taxonomic Classifier for 12S Fish Sequences

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Maintained by BioMac Lab](https://img.shields.io/badge/Maintained_by-BioMac_Lab-success?link=https%3A%2F%2Fwww.biomaclab.com)](https://www.biomaclab.com)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**AquaTaxa** is an advanced machine-learning tool designed for the rapid and accurate taxonomic classification of 12S rDNA sequences in ichthyology and environmental DNA (eDNA) studies. 

Developed by [BioMac Lab](https://www.biomaclab.com), AquaTaxa utilizes Deep Belief Networks (DBN) and Convolutional Neural Networks (CNN) to achieve high-fidelity taxonomic assignments, serving as a modern, deep-learning alternative to traditional k-mer or alignment-based classifiers like RDP and SINTAX.

## 🌟 Key Features
* **Deep Learning Architecture:** Utilizes optimized CNN and DBN models trained specifically on 12S rRNA fish datasets.
* **High Accuracy & Speed:** Designed to handle large-scale eDNA metabarcoding datasets efficiently.
* **Drop-in Alternative:** Can be easily integrated into existing bioinformatics pipelines requiring taxonomic assignment.

## 🚀 Quick Start

### 1. Installation
Clone the repository and install the required dependencies. We recommend using a virtual environment (like `conda` or `venv`).

```bash
git clone [https://github.com/biomaclab/AquaTaxa.git](https://github.com/biomaclab/AquaTaxa.git)
cd AquaTaxa
pip install -r requirements.txt