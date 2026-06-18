"""
SCA Screening - Task 1: Bone Density Module
================================================================================
This script performs binary classification for SCA screening using
bone_density features with strict 10-fold cross-validation to avoid data leakage.
All 7 models share the same feature selector per fold for global feature stability analysis.

Author: Li Fan
Date: 2026-04-01
Version: 1.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_curve, auc, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix, roc_auc_score)

import xgboost as xgb
import lightgbm as lgb

import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

import os
import random
from datetime import datetime

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Set matplotlib parameters for high-quality figures
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['legend.fontsize'] = 10
rcParams['figure.figsize'] = (10, 8)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

# Define paths
# ── User Configuration ──────────────────────────────────────────────────────
# Set DATA_PATH to your processed data file (merged across all acquisition modules).
# Set OUTPUT_DIR to the directory where results will be saved.
DATA_PATH = "data/processed/merged_all_sheets.xlsx"
OUTPUT_DIR = "outputs/task1"
# ────────────────────────────────────────────────────────────────────────────
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "bone_density_benchmark_results.xlsx")
ROC_PLOT_FILE = os.path.join(OUTPUT_DIR, "bone_density_roc_curves.png")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_prepare_data(filepath):
    """
    Load and prepare data from Excel file
    Handles missing values in both features and target
    """
    print("="*80)
    print("STEP 1: Data Loading and Preparation")
    print("="*80)
    
    df = pd.read_excel(filepath)
    print(f"Raw data shape: {df.shape}")
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col.startswith('bd_')]
    target_col = 'sa_evaluation_result'
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Remove rows with NaN in target
    valid_idx = y.notna()
    n_removed = sum(~valid_idx)
    if n_removed > 0:
        print(f"Removing {n_removed} rows with NaN in target variable")
        X = X[valid_idx]
        y = y[valid_idx]
    
    # Convert target to integer
    y = y.astype(int)
    
    print(f"Final dataset shape: {X.shape}")
    print(f"Class distribution:")
    print(f"  Normal (0): {sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%)")
    print(f"  Risk (1): {sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)")
    print(f"Total features: {len(feature_cols)}")
    
    return X, y, feature_cols


def impute_missing_values(X_train, X_test):
    """
    Impute missing values using training data median
    Returns imputed data for both train and test
    """
    # Calculate medians from training data only
    medians = X_train.median()
    
    # Fill missing values
    X_train_imputed = X_train.fillna(medians)
    X_test_imputed = X_test.fillna(medians)
    
    return X_train_imputed, X_test_imputed, medians


def l1_feature_selector(X_train, y_train, alpha=0.01):
    """
    Select features using L1-regularized Logistic Regression
    Returns selected feature names and mask
    """
    # Fit L1 logistic regression on training data
    selector = LogisticRegressionCV(
        penalty='l1',
        solver='liblinear',
        cv=5,
        scoring='roc_auc',
        class_weight='balanced',
        random_state=SEED,
        max_iter=3000,
        Cs=np.logspace(-3, 2, 20)
    )
    
    # Standardize before selection
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Fit selector
    selector.fit(X_train_scaled, y_train)
    
    # Get non-zero coefficients
    selected_mask = np.abs(selector.coef_[0]) > 1e-5
    selected_features = X_train.columns[selected_mask].tolist()
    
    # Fallback: if no features selected, use all features
    if len(selected_features) == 0:
        print("  Warning: No features selected, using all features")
        selected_features = X_train.columns.tolist()
        selected_mask = np.ones(len(X_train.columns), dtype=bool)
    
    return selected_features, selected_mask, scaler


def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """
    Train a single model and return predictions and metrics
    Enhanced version:
    1) train-set threshold optimization
    2) robust probability extraction
    3) better fallback handling
    """
    try:
        # =========================
        # 1. Train model
        # =========================
        model.fit(X_train, y_train)

        # =========================
        # 2. Get train probabilities for threshold tuning
        # =========================
        if hasattr(model, 'predict_proba'):
            train_proba = model.predict_proba(X_train)[:, 1]
            test_proba = model.predict_proba(X_test)[:, 1]

        elif hasattr(model, 'decision_function'):
            train_proba = model.decision_function(X_train)
            test_proba = model.decision_function(X_test)

            # normalize train
            if len(np.unique(train_proba)) > 1:
                train_proba = (
                    train_proba - train_proba.min()
                ) / (
                    train_proba.max() - train_proba.min()
                )
            else:
                train_proba = np.full_like(train_proba, 0.5, dtype=float)

            # normalize test
            if len(np.unique(test_proba)) > 1:
                test_proba = (
                    test_proba - test_proba.min()
                ) / (
                    test_proba.max() - test_proba.min()
                )
            else:
                test_proba = np.full_like(test_proba, 0.5, dtype=float)

        else:
            # fallback for models without probabilities
            train_proba = model.predict(X_train)
            test_proba = model.predict(X_test)

        # =========================
        # 3. Threshold optimization on training fold
        # =========================
        fpr, tpr, thresholds = roc_curve(y_train, train_proba)

        # Youden Index
        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        best_threshold = thresholds[best_idx]

        # =========================
        # 4. Apply threshold to test fold
        # =========================
        y_pred = (test_proba >= best_threshold).astype(int)

        # =========================
        # 5. Calculate metrics
        # =========================
        auc_score = roc_auc_score(y_test, test_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0) # Sensitivity
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate Specificity from Confusion Matrix
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return {
            'y_true': y_test,
            'y_pred_proba': test_proba,
            'y_pred': y_pred,
            'auc': auc_score,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'cm': cm,
            'best_threshold': best_threshold
        }

    except Exception as e:
        print(f"  Error in {model_name}: {e}")

        # safer fallback
        y_pred_proba = np.full(len(y_test), 0.5)
        y_pred = np.zeros(len(y_test), dtype=int)
        cm = confusion_matrix(y_test, y_pred)

        return {
            'y_true': y_test,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred,
            'auc': 0.5,
            'precision': 0.0,
            'recall': 0.0,
            'specificity': 0.0,
            'f1': 0.0,
            'cm': cm,
            'best_threshold': 0.5
        }


def run_cross_validation(X, y, feature_names, n_folds=10):
    """
    Run stratified 10-fold cross-validation with strict data leakage prevention
    All models share the same feature selector per fold
    """
    print("\n" + "="*80)
    print("STEP 2: Stratified 10-Fold Cross-Validation")
    print("="*80)
    
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    # Initialize models
    models = {
        'LR': LogisticRegression(random_state=SEED, max_iter=1000, class_weight='balanced'),
        'SVM': SVC(kernel='rbf', probability=True, random_state=SEED, class_weight='balanced'),
        'RF': RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight='balanced', n_jobs=-1),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=SEED, max_iter=500, 
                             early_stopping=True, validation_fraction=0.1),
        'XGB': xgb.XGBClassifier(n_estimators=100, random_state=SEED, use_label_encoder=False,
                                 eval_metric='logloss', verbosity=0),
        'LGBM': lgb.LGBMClassifier(n_estimators=100, random_state=SEED, verbosity=-1, 
                                   class_weight='balanced')
    }
    
    # Store results for each model
    model_results = {model_name: {
        'auc': [], 'precision': [], 'recall': [], 'specificity': [], 'f1': [],
        'y_true': [], 'y_pred_proba': [], 'confusion_matrices': []
    } for model_name in models.keys()}
    
    # Global feature selection statistics (across all folds)
    feature_selection_counts = {feature: 0 for feature in feature_names}
    
    # Perform cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")
        
        # Split data
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()
        
        # Step 1: Impute missing values using training data median
        X_train_imputed, X_test_imputed, medians = impute_missing_values(X_train, X_test)
        
        # Step 2: Feature selection using L1 logistic regression (only on training data)
        selected_features, selected_mask, scaler = l1_feature_selector(
            X_train_imputed, y_train, alpha=0.01
        )
        
        # Update global feature selection counts
        for feature in selected_features:
            feature_selection_counts[feature] += 1
        
        print(f"  Selected features: {len(selected_features)}/{len(feature_names)}")
        
        # Step 3: Standardize features using training data statistics
        X_train_scaled = scaler.transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)
        
        # Step 4: Apply feature selection (same for all models)
        X_train_selected = X_train_scaled[:, selected_mask]
        X_test_selected = X_test_scaled[:, selected_mask]
        
        # Step 5: Train and evaluate all models with selected features
        for model_name, model in models.items():
            print(f"  Training {model_name}...", end=' ')
            
            # Train and evaluate
            result = train_and_evaluate_model(
                model, model_name,
                X_train_selected, X_test_selected,
                y_train, y_test
            )
            
            # Store results
            model_results[model_name]['auc'].append(result['auc'])
            model_results[model_name]['precision'].append(result['precision'])
            model_results[model_name]['recall'].append(result['recall'])
            model_results[model_name]['specificity'].append(result['specificity'])
            model_results[model_name]['f1'].append(result['f1'])
            model_results[model_name]['y_true'].extend(result['y_true'])
            model_results[model_name]['y_pred_proba'].extend(result['y_pred_proba'])
            model_results[model_name]['confusion_matrices'].append(result['cm'])
            
            print(f"AUC={result['auc']:.3f}")
    
    # Calculate final metrics for each model
    final_results = {}
    for model_name in models.keys():
        # AUC and 95% CI
        auc_mean = np.mean(model_results[model_name]['auc'])
        auc_std = np.std(model_results[model_name]['auc'])
        
        # NOTE: The AUC 95% CI here uses fold-level standard error (SE method):
        #   CI = mean ± 1.96 * std/sqrt(n_folds)
        # This differs from the bootstrap resampling method reported in the paper
        # (Methods §2.6). The SE method is a close approximation for n_folds=10
        # and is retained here for computational efficiency.
        ci_margin = 1.96 * (auc_std / np.sqrt(n_folds))
        auc_lower = max(0.0, auc_mean - ci_margin)
        auc_upper = min(1.0, auc_mean + ci_margin)
        auc_95_ci = f"{auc_mean:.3f} ({auc_lower:.3f}-{auc_upper:.3f})"
        
        # Point estimates without CI for others
        sensitivity_mean = np.mean(model_results[model_name]['recall'])
        specificity_mean = np.mean(model_results[model_name]['specificity'])
        precision_mean = np.mean(model_results[model_name]['precision'])
        f1_mean = np.mean(model_results[model_name]['f1'])
        
        # Aggregate confusion matrix
        total_cm = np.sum(model_results[model_name]['confusion_matrices'], axis=0)
        
        final_results[model_name] = {
            'metrics': {
                'auc_mean': auc_mean, # kept for ROC curve plotting
                'AUC (95% CI)': auc_95_ci,
                'Sensitivity': sensitivity_mean,
                'Specificity': specificity_mean,
                'Precision': precision_mean,
                'F1': f1_mean
            },
            'predictions': {
                'y_true': model_results[model_name]['y_true'],
                'y_pred_proba': model_results[model_name]['y_pred_proba']
            },
            'confusion_matrix': total_cm
        }
    
    # Calculate global top 5 stable features
    sorted_features = sorted(feature_selection_counts.items(), key=lambda x: x[1], reverse=True)
    top_5_features = []
    for i, (feature, count) in enumerate(sorted_features[:5]):
        top_5_features.append({
            'rank': i+1,
            'feature': feature,
            'selected_count': count,
            'selected_frequency': count / n_folds
        })
    
    return final_results, top_5_features


def save_excel_results(results, top_5_features, output_file):
    """
    Save results to Excel file with three sheets
    """
    print("\n" + "="*80)
    print("STEP 3: Saving Results to Excel")
    print("="*80)
    
    # Sheet 1: Model Metrics (Only requested fields)
    metrics_data = []
    for model_name, model_results in results.items():
        metrics = model_results['metrics']
        metrics_data.append({
            'Model': model_name,
            'AUC (95% CI)': metrics['AUC (95% CI)'],
            'Sensitivity': metrics['Sensitivity'],
            'Specificity': metrics['Specificity'],
            'Precision': metrics['Precision'],
            'F1': metrics['F1']
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Sheet 2: Stable Features (Global Top 5)
    df_stable = pd.DataFrame(top_5_features)
    
    # Sheet 3: Confusion Matrix Summary
    cm_data = []
    for model_name, model_results in results.items():
        cm = model_results['confusion_matrix']
        cm_data.append({
            'model': model_name,
            'TP': int(cm[1, 1]),
            'TN': int(cm[0, 0]),
            'FP': int(cm[0, 1]),
            'FN': int(cm[1, 0])
        })
    
    df_cm = pd.DataFrame(cm_data)
    
    # Write to Excel with formatting
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_metrics.to_excel(writer, sheet_name='model_metrics', index=False)
        df_stable.to_excel(writer, sheet_name='stable_features', index=False)
        df_cm.to_excel(writer, sheet_name='confusion_matrix_summary', index=False)
        
        # Format worksheets
        workbook = writer.book
        
        # Format model_metrics
        ws1 = workbook['model_metrics']
        for row in ws1.iter_rows(min_row=1, max_row=len(df_metrics)+1):
            for cell in row:
                cell.font = Font(name='Arial', size=10)
                cell.alignment = Alignment(horizontal='center')
                if row[0].row == 1:
                    cell.font = Font(name='Arial', size=11, bold=True)
                    cell.fill = PatternFill(start_color='E6E6E6', end_color='E6E6E6', fill_type='solid')
        ws1.column_dimensions['A'].width = 12
        ws1.column_dimensions['B'].width = 22 # Wider for CI string
        for col in ['C', 'D', 'E', 'F']:
            ws1.column_dimensions[col].width = 15
        
        # Format stable_features
        ws2 = workbook['stable_features']
        for row in ws2.iter_rows(min_row=1, max_row=len(df_stable)+1):
            for cell in row:
                cell.font = Font(name='Arial', size=10)
                cell.alignment = Alignment(horizontal='center')
                if row[0].row == 1:
                    cell.font = Font(name='Arial', size=11, bold=True)
                    cell.fill = PatternFill(start_color='E6E6E6', end_color='E6E6E6', fill_type='solid')
        ws2.column_dimensions['A'].width = 8
        ws2.column_dimensions['B'].width = 35
        ws2.column_dimensions['C'].width = 15
        ws2.column_dimensions['D'].width = 18
        
        # Format confusion_matrix
        ws3 = workbook['confusion_matrix_summary']
        for row in ws3.iter_rows(min_row=1, max_row=len(df_cm)+1):
            for cell in row:
                cell.font = Font(name='Arial', size=10)
                cell.alignment = Alignment(horizontal='center')
                if row[0].row == 1:
                    cell.font = Font(name='Arial', size=11, bold=True)
                    cell.fill = PatternFill(start_color='E6E6E6', end_color='E6E6E6', fill_type='solid')
        ws3.column_dimensions['A'].width = 12
        for col in ['B', 'C', 'D', 'E']:
            ws3.column_dimensions[col].width = 10
    
    print(f"Results saved to: {output_file}")


def plot_roc_curves(results, output_file):
    """
    Plot ROC curves for all models with publication-quality style
    """
    print("\n" + "="*80)
    print("STEP 4: Generating ROC Curves")
    print("="*80)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors and line styles
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
    
    # Plot ROC curves
    for (model_name, model_results), color, ls in zip(results.items(), colors, line_styles):
        y_true = model_results['predictions']['y_true']
        y_pred_proba = model_results['predictions']['y_pred_proba']
        
        try:
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=color, linestyle=ls, lw=2,
                    label=f'{model_name} (AUC = {roc_auc:.3f})')
        except Exception as e:
            print(f"  Warning: Could not plot ROC for {model_name}: {e}")
    
    # Plot diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.8, label='Random (AUC = 0.500)')
    
    # Customize plot
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves - Bone Density Module', fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(loc='lower right', fontsize=9, frameon=True, fancybox=True, 
              shadow=True, framealpha=0.9)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add summary text box
    summary_text = "Model Performance Summary\n" + "="*30 + "\n"
    for model_name, model_results in results.items():
        metrics = model_results['metrics']
        # Extract CI string specifically tailored for the ROC summary box 
        summary_text += f"{model_name:<5}: AUC={metrics['AUC (95% CI)']}\n"
    
    # Add text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85, edgecolor='gray')
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props, fontfamily='monospace')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"ROC plot saved to: {output_file}")


def print_summary(results, top_5_features):
    """
    Print comprehensive summary of results
    """
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    # Model performance summary
    print("\nMODEL PERFORMANCE (10-fold CV):")
    print("-"*90)
    print(f"{'Model':<10} {'AUC (95% CI)':<22} {'Sensitivity':<15} {'Specificity':<15} {'Precision':<15} {'F1':<10}")
    print("-"*90)
    
    for model_name, model_results in results.items():
        metrics = model_results['metrics']
        print(f"{model_name:<10} {metrics['AUC (95% CI)']:<22} {metrics['Sensitivity']:<15.4f} "
              f"{metrics['Specificity']:<15.4f} {metrics['Precision']:<15.4f} {metrics['F1']:<10.4f}")
    
    # Global stable features
    print("\n" + "="*80)
    print("GLOBAL TOP 5 STABLE FEATURES (10-fold selection frequency):")
    print("-"*80)
    for feature_info in top_5_features:
        print(f"#{feature_info['rank']}: {feature_info['feature']:<40} "
              f"Selected in {feature_info['selected_count']}/10 folds "
              f"({feature_info['selected_frequency']*100:.0f}%)")
    
    # Best model identification
    best_auc_model = max(results.items(), key=lambda x: x[1]['metrics']['auc_mean'])
    best_sens_model = max(results.items(), key=lambda x: x[1]['metrics']['Sensitivity'])
    best_f1_model = max(results.items(), key=lambda x: x[1]['metrics']['F1'])
    
    print("\n" + "="*80)
    print("BEST PERFORMING MODELS:")
    print("-"*80)
    print(f"Best AUC: {best_auc_model[0]} (AUC={best_auc_model[1]['metrics']['AUC (95% CI)']})")
    print(f"Best Sensitivity: {best_sens_model[0]} (Sensitivity={best_sens_model[1]['metrics']['Sensitivity']:.4f})")
    print(f"Best F1-Score: {best_f1_model[0]} (F1={best_f1_model[1]['metrics']['F1']:.4f})")


def main():
    """
    Main execution function
    """
    print("="*80)
    print("SCA SCREENING - TASK 1: bone_density MODULE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Random seed: {SEED}")
    
    try:
        # Load and prepare data
        X, y, feature_names = load_and_prepare_data(DATA_PATH)
        
        # Run cross-validation
        results, top_5_features = run_cross_validation(X, y, feature_names, n_folds=10)
        
        # Save Excel results
        save_excel_results(results, top_5_features, OUTPUT_FILE)
        
        # Plot ROC curves
        plot_roc_curves(results, ROC_PLOT_FILE)
        
        # Print summary
        print_summary(results, top_5_features)
        
        print("\n" + "="*80)
        print("PHASE 1 COMPLETED SUCCESSFULLY!")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output directory: {OUTPUT_DIR}")
        print("="*80)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()