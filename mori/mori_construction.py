"""
SCA Screening - MoRI Construction
==================================
Constructs one Modular Risk Index (MoRI) per acquisition module.

Within each module:
  1. Near-constant features are removed.
  2. Missing values are imputed with the within-module median.
  3. Features are standardised with z-score normalisation.
  4. A fast Pearson correlation pre-screen retains the top 20 candidates.
  5. L1-regularised logistic regression (3-fold CV) selects and weights features.
  6. The MoRI is the resulting weighted linear combination.

The 20 MoRIs are saved to OUTPUT_DIR/mori_dataset_K20.xlsx, which is the
input expected by task3/task3_mori_ensemble.py.

Author: Li Fan
"""

import os
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel

warnings.filterwarnings("ignore")

# =====================================================
# Config
# =====================================================
# ── User Configuration ──────────────────────────────────────────────────────
# INPUT_FILE: processed data file with all acquisition modules merged.
# OUTPUT_DIR: directory where the MoRI dataset and feature report will be saved.
INPUT_FILE = "data/processed/merged_all_sheets.xlsx"
OUTPUT_DIR = "outputs/mori"
# ────────────────────────────────────────────────────────────────────────────

TARGET_COL = "sa_evaluation_result"

MODULE_PREFIXES = [
    "bsn_", "bsu_", "bsp_", "bsec_", "bscl_", "bsb_", "bsts_", "bf_","bsn2_",'bsu2_',"p_", "bc_", "bd_", "cr_", "sc_", "qdemo_", "qpa_", "qdbp_", "qpain_", "qshp_"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# Load data with optimization
# =====================================================
print("Loading data...")
df = pd.read_excel(INPUT_FILE)

df = df[df[TARGET_COL].notna()].copy()
y = (df[TARGET_COL] >= 1).astype(int)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if TARGET_COL in numeric_cols:
    numeric_cols.remove(TARGET_COL)

# =====================================================
# Optimized Module Risk Signature
# =====================================================
def process_module(prefix, df, numeric_cols, y):
    """Construct the MoRI for a single acquisition module (called in parallel)."""
    module_cols = [c for c in numeric_cols if c.startswith(prefix)]
    
    if len(module_cols) < 3:
        return None, None
    
    module_df = df[module_cols].copy()
    
    # Drop near-constant features
    nunique = module_df.nunique()
    high_var_cols = nunique[nunique > 2].index
    module_df = module_df[high_var_cols]
    
    if module_df.shape[1] < 3:
        return None, None
    
    # Median imputation
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(module_df)
    
    # Z-score standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Step 1: fast Pearson correlation pre-screen (top 20 features)
    correlations = np.array([np.corrcoef(X_scaled[:, i], y)[0, 1] for i in range(X_scaled.shape[1])])
    top_features = np.argsort(np.abs(correlations))[-min(20, len(correlations)):]
    
    if len(top_features) < 2:
        return None, None
    
    X_filtered = X_scaled[:, top_features]
    
    # Step 2: L1 logistic regression for final feature weighting
    try:
        selector = LogisticRegressionCV(
            penalty="l1",
            solver="liblinear",
            cv=3,
            scoring="roc_auc",
            max_iter=1000,
            random_state=42,
            n_jobs=1  # Avoid nested parallelism
        )
        selector.fit(X_filtered, y)
        
        coef = selector.coef_.flatten()
        selected_mask = coef != 0
        
        if selected_mask.sum() == 0:
            # Fallback: use the single most-correlated feature
            selected_mask = np.zeros_like(coef, dtype=bool)
            selected_mask[np.argmax(np.abs(correlations[top_features]))] = True
            coef = np.ones_like(coef)
        
        selected_features = module_df.columns[top_features[selected_mask]]
        selected_coef = coef[selected_mask]
        
        # Compute the MoRI as a weighted linear combination
        module_score = np.dot(X_filtered[:, selected_mask], selected_coef)
        
        # Compile feature-level report for this module
        reports = []
        for feat, c in zip(selected_features, selected_coef):
            reports.append({
                "module": prefix,
                "selected_feature": feat,
                "coefficient": c
            })
        
        return module_score, reports
        
    except Exception as e:
print(f"Module {prefix} failed: {str(e)}")
        return None, None

# Parallel MoRI construction for all modules
print("Constructing MoRIs for all modules in parallel...")
results = Parallel(n_jobs=-1, verbose=10)(
    delayed(process_module)(prefix, df, numeric_cols, y) 
    for prefix in MODULE_PREFIXES
)

# Collect results
risk_dataset = pd.DataFrame(index=df.index)
module_reports = []

for module_score, reports in results:
    if module_score is not None and reports is not None:
        # Generate unique column name
        base_name = f"{reports[0]['module']}risk_score"
        col_name = base_name
        counter = 1
        while col_name in risk_dataset.columns:
            col_name = f"{base_name}_{counter}"
            counter += 1
        risk_dataset[col_name] = module_score
        module_reports.extend(reports)

# =====================================================
# Post-processing: remove highly correlated MoRI scores
# =====================================================
print("Post-processing: removing redundant MoRI scores...")
if len(risk_dataset.columns) > 0:
    # Compute pairwise correlation matrix
    corr_matrix = risk_dataset.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Drop one of each pair with |r| > 0.95
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    risk_dataset = risk_dataset.drop(columns=to_drop)
    print(f"Removed {len(to_drop)} highly correlated MoRI score(s)")

# =====================================================
# Save outputs
# =====================================================
risk_dataset[TARGET_COL] = y.values

risk_output = os.path.join(OUTPUT_DIR, "mori_dataset_K20.xlsx")
report_output = os.path.join(OUTPUT_DIR, "mori_feature_report.xlsx")

print("Saving outputs...")
risk_dataset.to_excel(risk_output, index=False)
pd.DataFrame(module_reports).to_excel(report_output, index=False)

print(f"MoRI dataset saved to: {risk_output}")
print(f"Module feature report saved to: {report_output}")
print(f"Number of MoRI scores generated: {len(risk_dataset.columns)-1}")

# =====================================================
# Quick baseline training with early stopping
# =====================================================
if len(risk_dataset) > 0 and len(risk_dataset.columns) > 1:
    print("\nRunning 10-fold cross-validation on the MoRI dataset...")
    X_risk = risk_dataset.drop(columns=[TARGET_COL])
    
    # Dimensionality reduction guard (should not be needed for ≤20 MoRIs)
    if X_risk.shape[1] > 50:
        print(f"Feature count ({X_risk.shape[1]}) exceeds 50; applying PCA.")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(50, X_risk.shape[0]//2))
        X_risk = pd.DataFrame(pca.fit_transform(X_risk))
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    aucs = []
    
    for fold, (tr_idx, te_idx) in enumerate(cv.split(X_risk, y)):
        X_tr = X_risk.iloc[tr_idx]
        X_te = X_risk.iloc[te_idx]
        y_tr = y.iloc[tr_idx]
        y_te = y.iloc[te_idx]
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_tr, y_tr)
        
        pred = model.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, pred)
        aucs.append(auc)
        print(f"Fold {fold+1}/10: AUC = {auc:.4f}")
    
    print(f"\nMoRI baseline 10-fold AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
else:
    print("\nWARNING: No valid MoRI scores were generated; skipping cross-validation.")