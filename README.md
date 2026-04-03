# Hybrid Metaheuristic Feature Selection for Mammography

This repository contains the complete, ready-to-run Python experimental codebase that implements the full pipeline described in the manuscript:

> **“Hybrid Metaheuristic Feature Selection for Enhanced Breast Cancer Detection in Digital Mammography: A Radiomics and Deep Learning Approach with Cross-Dataset Validation”**

## Overview

The pipeline is designed to be modular and reproducible. It covers:
1. **Data Loading**: Parsing metadata for CBIS-DDSM and VinDr-Mammo.
2. **Preprocessing**: DICOM loading, intensity normalization, and ROI extraction.
3. **Feature Extraction**: IBSI-compliant radiomics (`pyradiomics`) and deep features (EfficientNet-B5).
4. **Feature Selection**: A novel hybrid Grasshopper Optimization Algorithm + Crow Search Algorithm (GOA-CSA).
5. **Classification**: An MLP classifier trained on the selected feature subset.
6. **Evaluation**: Rigorous internal (CBIS-DDSM) and external (VinDr-Mammo) validation, including ROC analysis and generation of all manuscript tables and figures.

## Project Structure

```text
/mammography_pipeline
├── config/                 # Central configuration files
│   ├── config.py           # Main config (paths, hyperparameters)
│   └── pyradiomics_params.yaml  # Radiomics feature settings
├── data/                   # Data loading and preprocessing modules
│   ├── load_cbis_ddsm.py
│   ├── load_vindr.py
│   └── preprocessing.py
├── features/               # Feature extraction and merging
│   ├── extract_radiomics.py
│   ├── extract_deep.py
│   └── merge_features.py
├── selection/              # Hybrid GOA-CSA feature selection
│   └── goa_csa.py
├── models/                 # MLP classifier model and training
│   └── mlp_model.py
├── evaluation/             # Metrics calculation and figure generation
│   └── evaluate.py
├── outputs/                # Default directory for all generated files
│   ├── figures/            # ROC curves, flowcharts
│   ├── results/            # Feature CSVs, model checkpoints, metrics
│   └── rois/               # Saved ROI image patches (PNG)
├── main.py                 # Main pipeline orchestration script
└── README.md               # This file
```

## Setup

### 1. Dependencies

It is recommended to use a Python 3.10+ virtual environment.

```bash
pip install numpy pandas scikit-learn pydicom pillow torch torchvision timm SimpleITK pyradiomics matplotlib
```

### 2. Datasets

Download datasets from the official sources:

- **CBIS-DDSM**: The Cancer Imaging Archive (TCIA)  
  https://www.cancerimagingarchive.net/collection/cbis-ddsm/

- **VinDr-Mammo**: PhysioNet  
  https://physionet.org/content/vindr-mammo/1.0.0/

Configure paths by editing `config/config.py`:

- `CBIS_DDSM_IMAGE_DIR`
- `CBIS_DDSM_MASK_DIR`
- `CBIS_DDSM_CSV`
- `VINDR_IMAGE_DIR`
- `VINDR_FINDING_CSV`

## Usage

The `main.py` script orchestrates the entire pipeline. You can run the full pipeline from scratch or execute specific steps using command-line flags.

### Running the full pipeline

To run all steps from data loading to final evaluation:

```bash
python3.11 main.py --all
```

This will take significant time, especially during feature extraction and selection.

### Running specific steps

You can run a subset of the pipeline by combining flags. Steps will execute in logical order.

**Example 1: Feature extraction, selection, training, and evaluation**  
(Assumes data has already been loaded and preprocessed)

```bash
python3.11 main.py --extract --merge --select --train --evaluate
```

**Example 2: Re-run only the final evaluation**  
(Assumes features have been extracted, models trained, and selection mask saved)

```bash
python3.11 main.py --evaluate
```

### Command-line flags

- `--all`       : Run all steps from scratch.  
- `--load`      : Step 1 – Data loading & preprocessing.  
- `--extract`   : Step 2 – Feature extraction (radiomics + deep).  
- `--finetune`  : Optional fine-tuning of EfficientNet-B5 during deep feature extraction (requires GPU).  
- `--merge`     : Step 3 – Feature merging.  
- `--select`    : Step 4 – Hybrid GOA-CSA feature selection.  
- `--train`     : Step 5 – MLP model training.  
- `--evaluate`  : Step 6 – Final evaluation and figure generation.  

## Outputs

All generated files are saved under `outputs/` by default:

- `outputs/rois/`  
  Preprocessed ROI image patches (16-bit PNGs).

- `outputs/results/`  
  - `preprocessed_*.csv` – Metadata for preprocessed ROIs.  
  - `radiomics_features.csv` – Extracted radiomic features.  
  - `deep_features.csv` – Extracted deep features.  
  - `combined_features.csv` – Merged feature matrix.  
  - `selected_feature_mask.npy` – Binary mask from GOA-CSA.  
  - `mlp_*.pt` – Trained PyTorch model checkpoints.  
  - `metrics_summary.csv` – Final table of performance metrics.

- `outputs/figures/`  
  - `figure1_roc_curves.png` – Publication-quality ROC curve figure.

## Reproducing manuscript results

To reproduce the main results reported in the manuscript (Tables 1–2 and Figure 1):

```bash
python3.11 main.py --all
```

After completion, the key outputs will be:

- `outputs/results/metrics_summary.csv` – Internal and external performance metrics.  
- `outputs/figures/figure1_roc_curves.png` – ROC curves for CBIS-DDSM and VinDr-Mammo.

## Citation

If you use this repository in academic work, please cite:

> Alshreef BS, et al. Hybrid Metaheuristic Feature Selection for Enhanced Breast Cancer Detection in Digital Mammography: A Radiomics and Deep Learning Approach with Cross-Dataset Validation. [Journal name, year – to be updated after publication].

## Citation and archival record

A versioned snapshot of this repository is archived on Zenodo.

- Version-specific DOI: `10.5281/zenodo.19401611`
- Concept DOI (all versions): `10.5281/zenodo.19401610`

Certain implementation aspects of this framework are related to a filed Saudi patent application by the author (Patent Application No. SA1020262200, filed on 17 March 2026). The application is currently awaiting formal examination.  
