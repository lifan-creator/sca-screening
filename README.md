# sca-screening

Code repository for:

> **Development and Validation of Modular Risk Indices for Multimodal Spinal Curvature Abnormality Screening in Young Adults**

---

## Overview

This repository contains the complete Python pipeline for the SCA (Spinal Curvature Abnormality) screening study. The pipeline has three sequential tasks that benchmark progressively richer multimodal fusion strategies, plus a cost-aware deployment analysis:

| Task | Directory | Script | Description |
|------|-----------|--------|-------------|
| MoRI construction | `mori/` | `mori_construction.py` | Build one Modular Risk Index (MoRI) per acquisition module |
| Task 1 | `task1/` | `balance.py`, `body_composition.py`, … | Single-module baseline models (11 scripts, one per module) |
| Task 2 | `task2/` | `naive_multimodal_concat.py` | Naïve multi-module concatenation with stability-driven feature selection |
| Task 3 | `task3/` | `mori_ensemble.py` | Soft-voting MoRI ensemble with Kernel SHAP interpretability |
| Deployment | `deployment/` | `deployment_K3/5/7/8.py` | Cost-aware reduced configurations (K = 3, 5, 7, 8 MoRIs) |

---

## Repository structure

```
sca-screening/
├── data/
│   └── data-dictionary.xlsx        # Variable descriptions (see paper Supplementary S1)
├── mori/
│   └── mori_construction.py        # Step 1 — construct the full 20-MoRI dataset
├── task1/                          # Task 1 — single-module baselines
│   ├── balance.py
│   ├── body_composition.py
│   ├── bone_density.py
│   ├── cervical_rom.py
│   ├── postural_alignment.py
│   ├── spinal_curvature.py
│   ├── q_daily_behavioral_posture.py
│   ├── q_demographics.py
│   ├── q_pain_assessment.py
│   ├── q_physical_activity.py
│   └── q_spine_health_perception.py
├── task2/
│   └── naive_multimodal_concat.py  # Task 2 — naïve concatenation
├── task3/
│   └── mori_ensemble.py            # Task 3 — MoRI ensemble + SHAP
├── deployment/
│   ├── deployment_K3.py            # Deployment: top-3 MoRIs (minimum-viable)
│   ├── deployment_K5.py            # Deployment: top-5 MoRIs
│   ├── deployment_K7.py            # Deployment: top-7 MoRIs
│   └── deployment_K8.py            # Deployment: top-8 MoRIs (recommended)
├── preprocessing/
│   └── README.md                   # Preprocessing steps (see paper Methods §2.4)
├── requirements.txt
└── README.md
```

---

## Data

Raw participant data are not publicly released in compliance with the study ethics approval (Tongji University Science and Technology Ethics Committee, No. 2021tjdx025). The data dictionary describing all variables is provided in `data/data-dictionary.xlsx`. Researchers may request access to de-identified data by contacting the corresponding author.

The pipeline expects a single merged Excel file with one row per participant and column names matching the prefixes listed in `mori/mori_construction.py` (`MODULE_PREFIXES`).

---

## Environment setup

Python 3.9 or later is recommended.

```bash
pip install -r requirements.txt
```

---

## Running the pipeline

Scripts must be run in the order below. Update the `DATA_PATH` / `INPUT_FILE` and `OUTPUT_DIR` variables at the top of each script before running.

### Step 1 — Construct MoRIs

```bash
python mori/mori_construction.py
```

Output: `outputs/mori/mori_dataset_K20.xlsx` (full 20-MoRI dataset, one row per participant)

### Step 2 — Task 1: single-module baselines

Run any or all of the 11 scripts in `task1/`. Each script trains seven classifiers on a single acquisition module using stratified 10-fold cross-validation.

```bash
python task1/balance.py
python task1/cervical_rom.py
# … repeat for other modules as needed
```

### Step 3 — Task 2: naïve multimodal concatenation

```bash
python task2/naive_multimodal_concat.py
```

### Step 4 — Task 3: MoRI ensemble

```bash
python task3/mori_ensemble.py
```

Input: `outputs/mori/mori_dataset_K20.xlsx`  
Output: results Excel + ROC curve + SHAP plots in `outputs/task3/`

### Step 5 — Deployment configurations (K-reduced)

For each reduced configuration, first generate the corresponding K-MoRI dataset by selecting the top-K SHAP-ranked MoRI columns from `mori_dataset_K20.xlsx` (ranked by mean |SHAP| from the Task 3 output) and saving it under the expected filename:

| Script | Input file | MoRIs used |
|--------|-----------|-----------|
| `deployment_K3.py` | `outputs/mori/mori_dataset_K3.xlsx` | top-3 SHAP-ranked |
| `deployment_K5.py` | `outputs/mori/mori_dataset_K5.xlsx` | top-5 SHAP-ranked |
| `deployment_K7.py` | `outputs/mori/mori_dataset_K7.xlsx` | top-7 SHAP-ranked |
| `deployment_K8.py` | `outputs/mori/mori_dataset_K8.xlsx` | top-8 SHAP-ranked |

```bash
python deployment/deployment_K3.py
python deployment/deployment_K8.py
# … etc.
```

---

## Reproducibility

All scripts use `RANDOM_STATE = 42`. Results are obtained with stratified 10-fold cross-validation. The AUC 95% CI is computed via the fold-level standard-error method (a close approximation to the bootstrap CI reported in the paper for n_folds = 10).

---

## Citation

If you use this code, please cite the paper:

> Li Fan et al. "Development and Validation of Modular Risk Indices for Multimodal Spinal Curvature Abnormality Screening in Young Adults." *(Journal name, year, DOI.)*

---

## License

This code is released for academic research use. Please contact the authors before any commercial use.
