"""
SCA Screening - Preprocessing: Data Quality Audit
================================================================================
Comprehensive quality audit of the merged dataset prior to modelling.

Audit modules:
  1. Structural QA     — sample/feature counts, constant cols, dtype issues
  2. Missingness       — per-column and per-sample rates, module-level summary,
                         MAR/MNAR mechanism proxy via chi-square test
  3. Outliers          — IQR, z-score, Isolation Forest, Local Outlier Factor
  4. Label QA          — class distribution, suspected label noise via CV
  5. Duplicates        — exact and near-duplicate detection
  6. Collinearity      — highly correlated feature pairs (|r| > 0.90)
  7. Re-cleaning plan  — prioritised action list with expected impact

Output:
    data_check_report.xlsx  (one sheet per audit module)

Author: Li Fan
Date: 2026-01-22
Version: 2.0
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from scipy.stats import zscore, chi2_contingency
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import cross_val_predict, StratifiedKFold


# =========================================================
# CONFIG
# =========================================================
INPUT_FILE = r"D:\科研\ais_multimodal_multitask_pipeline\data\processed\merged_all_sheets.xlsx"
OUTPUT_FILE = r"data_check_report.xlsx"
TARGET_COL = "sa_evaluation_result"


# =========================================================
# LOAD
# =========================================================
print("Loading data...")
df = pd.read_excel(INPUT_FILE)
print(f"Loaded: {df.shape}")


# =========================================================
# BASIC TYPE SPLIT
# =========================================================
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if TARGET_COL in numeric_cols:
    numeric_cols.remove(TARGET_COL)

object_cols = df.select_dtypes(include="object").columns.tolist()


# =========================================================
# 1 STRUCTURAL QUALITY
# =========================================================
print("Checking structure...")

n_samples, n_features = df.shape
duplicate_rows = df.duplicated().sum()

id_candidates = [c for c in df.columns if "id" in c.lower()]
id_duplicate_records = []
for c in id_candidates:
    id_duplicate_records.append({
        "id_column": c,
        "duplicate_count": int(df[c].duplicated().sum())
    })

constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

near_zero_var_cols = []
for c in numeric_cols:
    vc = df[c].value_counts(normalize=True, dropna=False)
    if len(vc) > 1 and vc.iloc[0] > 0.95:
        near_zero_var_cols.append(c)

dtype_issues = []
for c in object_cols:
    sample = df[c].dropna().astype(str).head(20)
    try:
        pd.to_numeric(sample)
        dtype_issues.append(c)
    except:
        pass

spelling_inconsistency = []
for c in object_cols:
    nunique_raw = df[c].dropna().nunique()
    normalized = df[c].dropna().astype(str).str.strip().str.lower()
    nunique_norm = normalized.nunique()
    if nunique_norm < nunique_raw:
        spelling_inconsistency.append(c)

structure_df = pd.DataFrame({
    "Metric": [
        "n_samples",
        "n_features",
        "duplicate_rows",
        "target_missing_count",
        "target_missing_rate",
        "constant_col_count",
        "near_zero_var_count",
        "dtype_issue_count",
        "spelling_inconsistency_count"
    ],
    "Value": [
        n_samples,
        n_features,
        duplicate_rows,
        int(df[TARGET_COL].isna().sum()) if TARGET_COL in df.columns else 0,
        round(df[TARGET_COL].isna().mean(), 4) if TARGET_COL in df.columns else 0,
        len(constant_cols),
        len(near_zero_var_cols),
        len(dtype_issues),
        len(spelling_inconsistency)
    ]
})


# =========================================================
# 2 MISSINGNESS
# =========================================================
print("Checking missingness...")

col_missing_df = (
    df.isna()
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)
col_missing_df.columns = ["feature", "missing_rate"]

sample_missing_df = pd.DataFrame({
    "sample_index": np.arange(len(df)),
    "missing_rate": df.isna().mean(axis=1)
})

# module missingness by prefix
module_records = []
module_map = {}
for c in df.columns:
    prefix = c.split("_")[0]
    module_map.setdefault(prefix, []).append(c)

for m, cols in module_map.items():
    module_records.append({
        "module": m,
        "feature_count": len(cols),
        "missing_rate": round(df[cols].isna().mean().mean(), 4)
    })

module_missing_df = pd.DataFrame(module_records).sort_values(
    "missing_rate", ascending=False
)

# MAR / MNAR proxy
missing_label_relation = []
if TARGET_COL in df.columns:
    for col in df.columns:
        if col == TARGET_COL:
            continue
        try:
            tmp = df[[col, TARGET_COL]].copy()
            tmp = tmp[tmp[TARGET_COL].notna()]
            tab = pd.crosstab(tmp[col].isna(), tmp[TARGET_COL])

            if tab.shape[0] == 2 and tab.shape[1] >= 2:
                chi2, p, _, _ = chi2_contingency(tab)
                miss_rate = df[col].isna().mean()

                mech = "MCAR_like"
                if p < 0.05:
                    mech = "MAR_or_MNAR"
                if miss_rate > 0.4 and p < 0.05:
                    mech = "MNAR_high_risk"

                missing_label_relation.append({
                    "feature": col,
                    "missing_rate": round(miss_rate, 4),
                    "p_value": round(p, 6),
                    "mechanism_proxy": mech
                })
        except:
            continue

missing_mech_df = pd.DataFrame(missing_label_relation)


# =========================================================
# 3 OUTLIERS
# =========================================================
print("Checking outliers...")

outlier_records = []
iso_outlier_count = 0
lof_outlier_count = 0

if len(numeric_cols) > 0:
    num_df = df[numeric_cols].copy()
    num_df = num_df.fillna(num_df.median())

    for col in numeric_cols:
        s = num_df[col]

        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        iqr_n = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()

        z_n = (np.abs(zscore(s)) > 3).sum()

        outlier_records.append({
            "feature": col,
            "iqr_outliers": int(iqr_n),
            "zscore_outliers": int(z_n)
        })

    iso = IsolationForest(contamination=0.03, random_state=42)
    iso_flag = iso.fit_predict(num_df)
    iso_outlier_count = int((iso_flag == -1).sum())

    lof = LocalOutlierFactor(
        n_neighbors=min(20, len(num_df)-1),
        contamination=0.03
    )
    lof_flag = lof.fit_predict(num_df)
    lof_outlier_count = int((lof_flag == -1).sum())

outlier_df = pd.DataFrame(outlier_records)

outlier_summary_df = pd.DataFrame({
    "method": ["IsolationForest", "LOF"],
    "outlier_count": [iso_outlier_count, lof_outlier_count]
})


# =========================================================
# 4 LABEL QA
# =========================================================
print("Checking label quality...")

label_dist_df = pd.DataFrame()
suspected_label_noise_df = pd.DataFrame()

if TARGET_COL in df.columns:
    label_dist_df = (
        df[TARGET_COL]
        .value_counts(dropna=False)
        .reset_index()
    )
    label_dist_df.columns = ["class", "count"]

    label_df = df[df[TARGET_COL].notna()].copy()

    if len(label_df) >= 30:
        class_counts = label_df[TARGET_COL].value_counts()
        valid_classes = class_counts[class_counts >= 5].index
        label_df = label_df[label_df[TARGET_COL].isin(valid_classes)]

        if len(label_df) >= 30 and label_df[TARGET_COL].nunique() >= 2:
            X = label_df[numeric_cols].copy()
            X = X.fillna(X.median())
            y = label_df[TARGET_COL]

            min_class = y.value_counts().min()
            n_splits = min(5, min_class)

            if n_splits >= 2:
                clf = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=5,
                    random_state=42,
                    class_weight="balanced"
                )

                cv = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=42
                )

                pred = cross_val_predict(clf, X, y, cv=cv)

                bad_idx = np.where(pred != y)[0]
                suspected_label_noise_df = label_df.iloc[bad_idx].copy()
                suspected_label_noise_df["predicted_label"] = pred[bad_idx]


# =========================================================
# 5 DUPLICATE / NEAR DUPLICATE
# =========================================================
print("Checking duplicates...")

duplicate_df = df[df.duplicated(keep=False)].copy()

near_dup_df = pd.DataFrame()
if len(numeric_cols) >= 3:
    rounded = df[numeric_cols].round(2)
    near_mask = rounded.duplicated(keep=False)
    near_dup_df = df.loc[near_mask].copy()


# =========================================================
# 6 COLLINEARITY
# =========================================================
print("Checking collinearity...")

corr_records = []
if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    for col in upper.columns:
        rows = upper.index[upper[col] > 0.9].tolist()
        for r in rows:
            corr_records.append({
                "feature_1": r,
                "feature_2": col,
                "corr": round(upper.loc[r, col], 4)
            })

corr_df = pd.DataFrame(corr_records)


# =========================================================
# 7 RE-CLEAN ACTIONS
# =========================================================
print("Generating reclean actions...")

actions = []

if duplicate_rows > 0:
    actions.append(["P0", "Remove exact duplicate samples", "High"])

if TARGET_COL in df.columns and df[TARGET_COL].isna().mean() > 0:
    actions.append(["P0", "Remove unlabeled samples for supervised modeling", "High"])

if len(suspected_label_noise_df) > 0:
    actions.append(["P0", "Manual review suspected label noise", "Very High"])

if len(missing_mech_df) > 0:
    high_risk = (missing_mech_df["mechanism_proxy"] == "MNAR_high_risk").sum()
    if high_risk > 0:
        actions.append(["P1", "Use missing indicator + iterative imputation", "High"])

if len(corr_df) > 20:
    actions.append(["P1", "Remove highly collinear features", "Medium"])

if len(near_zero_var_cols) > 0:
    actions.append(["P1", "Remove near-zero variance features", "Medium"])

if len(outlier_df) > 0:
    actions.append(["P1", "Create anomaly_flag instead of direct deletion", "Medium"])

reclean_df = pd.DataFrame(
    actions,
    columns=["Priority", "Action", "Expected_Gain"]
)


# =========================================================
# QUALITY SCORE
# =========================================================
score = 100
score -= min(15, duplicate_rows)
score -= min(15, len(constant_cols))
score -= min(15, len(near_zero_var_cols))
score -= min(20, len(corr_df) // 10)
score -= min(20, len(suspected_label_noise_df) // 10)

if TARGET_COL in df.columns:
    score -= min(15, int(df[TARGET_COL].isna().mean() * 100))

score = max(score, 0)

summary_df = pd.DataFrame({
    "Metric": ["Overall_Quality_Score"],
    "Value": [score]
})


# =========================================================
# EXPORT
# =========================================================
print("Exporting Excel report...")

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    structure_df.to_excel(writer, sheet_name="Structure", index=False)
    col_missing_df.to_excel(writer, sheet_name="Missingness", index=False)
    module_missing_df.to_excel(writer, sheet_name="Module_Missing", index=False)
    sample_missing_df.to_excel(writer, sheet_name="Sample_Missing", index=False)
    missing_mech_df.to_excel(writer, sheet_name="Missing_Mechanism", index=False)
    outlier_df.to_excel(writer, sheet_name="Outliers", index=False)
    outlier_summary_df.to_excel(writer, sheet_name="Outlier_Summary", index=False)
    label_dist_df.to_excel(writer, sheet_name="Label_QA", index=False)
    suspected_label_noise_df.head(1000).to_excel(
        writer, sheet_name="Suspected_Label_Noise", index=False
    )
    duplicate_df.head(1000).to_excel(writer, sheet_name="Duplicate", index=False)
    near_dup_df.head(1000).to_excel(writer, sheet_name="Near_Duplicate", index=False)
    corr_df.to_excel(writer, sheet_name="Collinearity", index=False)
    reclean_df.to_excel(writer, sheet_name="Reclean_Actions", index=False)

print(f"\n✅ Report saved successfully: {OUTPUT_FILE}")