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
