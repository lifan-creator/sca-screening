"""
SCA Screening - Preprocessing: Merged Dataset Secondary Cleaner
================================================================================
Secondary cleaning pass applied after data_quality_audit.py has been reviewed.
Implements the prioritised actions from the audit report.

Pipeline steps:
  1. Remove unlabelled samples (missing target column)
  2. Create module-level missingness flags (bd_, bc_, q_ prefixes)
  3. Module-wise KNN imputation for numeric features
  4. Global median fill for any remaining numeric NaNs
  5. Anomaly flag via Isolation Forest
  6. Drop highly collinear features (|r| > 0.95)
  7. Drop constant and near-zero-variance columns (dominant category > 98%)

Input:  merged_all_sheets.xlsx
Output: cleaned_all_sheets.xlsx

Author: Li Fan
Date: 2026-04-09
Version: 3.0
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest


# =====================================================
# CONFIG
# =====================================================
INPUT_FILE = r"D:\科研\ais_multimodal_multitask_pipeline\data\processed\merged_all_sheets.xlsx"
OUTPUT_FILE = r"D:\科研\ais_multimodal_multitask_pipeline\data\processed\cleaned_all_sheets.xlsx"
TARGET_COL = "sa_evaluation_result"


# =====================================================
# LOAD
# =====================================================
print("Loading data...")
df = pd.read_excel(INPUT_FILE)
print(f"Original shape: {df.shape}")


# =====================================================
# STEP 1: REMOVE UNLABELED SAMPLES
# =====================================================
print("Step 1: removing unlabeled samples...")
if TARGET_COL in df.columns:
    before = len(df)
    df = df[df[TARGET_COL].notna()].copy()
    after = len(df)
    print(f"Removed {before - after} unlabeled samples")
    print(f"Current shape: {df.shape}")


# =====================================================
# STEP 2: MODULE-LEVEL MISSING FLAGS
# =====================================================
print("Step 2: creating module missing flags...")

module_prefixes = ["bd", "bc", "q"]

for prefix in module_prefixes:
    cols = [c for c in df.columns if c.startswith(prefix + "_")]
    if len(cols) > 0:
        df[f"{prefix}_missing_flag"] = (
            df[cols].isna().mean(axis=1) > 0.5
        ).astype(int)


# =====================================================
# STEP 3: MODULE-WISE KNN IMPUTATION
# =====================================================
print("Step 3: module-wise imputation...")

for prefix in module_prefixes:
    cols = [c for c in df.columns if c.startswith(prefix + "_")]
    if len(cols) == 0:
        continue

    numeric_cols = df[cols].select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) >= 2:
        print(f"Imputing module: {prefix} ({len(numeric_cols)} cols)")
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])


# =====================================================
# STEP 4: GLOBAL NUMERIC SAFE IMPUTATION
# =====================================================
print("Step 4: filling remaining numeric missing...")

numeric_cols_all = df.select_dtypes(include=np.number).columns.tolist()

for col in numeric_cols_all:
    if df[col].isna().sum() > 0:
        df[col] = df[col].fillna(df[col].median())


# =====================================================
# STEP 5: ANOMALY FLAG
# =====================================================
print("Step 5: creating anomaly flag...")

feature_cols = [
    c for c in numeric_cols_all
    if c != TARGET_COL
]

if len(feature_cols) >= 5:
    iso = IsolationForest(
        contamination=0.03,
        random_state=42
    )

    anomaly_pred = iso.fit_predict(df[feature_cols])
    df["anomaly_flag"] = (anomaly_pred == -1).astype(int)
else:
    df["anomaly_flag"] = 0


# =====================================================
# STEP 6: HIGH COLLINEARITY FILTER
# =====================================================
print("Step 6: removing highly collinear features...")

numeric_model_cols = [
    c for c in df.select_dtypes(include=np.number).columns
    if c != TARGET_COL
]

corr = df[numeric_model_cols].corr().abs()

upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

drop_cols = [
    col for col in upper.columns
    if any(upper[col] > 0.95)
]

print(f"Removing {len(drop_cols)} highly collinear columns")
df = df.drop(columns=drop_cols)


# =====================================================
# STEP 7: REMOVE CONSTANT / NEAR ZERO VAR
# =====================================================
print("Step 7: removing constant and near-zero variance cols...")

remove_cols = []

for c in df.columns:
    nunique = df[c].nunique(dropna=False)
    if nunique <= 1:
        remove_cols.append(c)
        continue

    if pd.api.types.is_numeric_dtype(df[c]):
        vc = df[c].value_counts(normalize=True, dropna=False)
        if len(vc) > 1 and vc.iloc[0] > 0.98:
            remove_cols.append(c)

df = df.drop(columns=list(set(remove_cols)), errors="ignore")
print(f"Removed {len(set(remove_cols))} low-information columns")


# =====================================================
# STEP 8: SAVE
# =====================================================
print("Saving cleaned data...")

df.to_excel(OUTPUT_FILE, index=False)

print(f"\n✅ Cleaned dataset saved to:\n{OUTPUT_FILE}")
print(f"Final shape: {df.shape}")