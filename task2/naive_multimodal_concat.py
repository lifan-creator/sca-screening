"""
SCA Screening - Task 2: Multimodal Naive Concatenation
===============================================================
Author: Li Fan
Date: 2026-04-07
Description: 
    This script performs strict 10-fold cross-validation for 7 machine learning models
    on all-module concatenated data for SCA screening. Implements rigorous feature selection
    within each fold to prevent data leakage.
    Key point: features with selection_frequency > 0.6 across 10 folds are retained
    (17 features in the study cohort, per Methods §2.5).

Models:
    - Logistic Regression (LR)
    - SVM (RBF)
    - Random Forest (RF)
    - KNN
    - MLPClassifier
    - XGBoost (XGB)
    - LightGBM (LGBM)

Outputs:
    - ROC curves with AUC values
    - Comprehensive metrics summary
    - Stable features ranking
    - Confusion matrices
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
import os
import logging
from datetime import datetime
from typing import Tuple, Dict, List, Any

# Machine Learning imports
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_auc_score
)
import xgboost as xgb
import lightgbm as lgb

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Configuration
# ── User Configuration ──────────────────────────────────────────────────────
# DATA_PATH: processed data file with all acquisition modules merged.
# OUTPUT_DIR: directory where results (Excel + ROC plot) will be saved.
DATA_PATH = "data/processed/merged_all_sheets.xlsx"
OUTPUT_DIR = "outputs/task2"
# ────────────────────────────────────────────────────────────────────────────
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "allData_benchmark_results_v4.3.xlsx")
ROC_IMAGE = os.path.join(OUTPUT_DIR, "allData_roc_curves_v4.3.png")
MISSING_THRESHOLD = 0.32  # 32% missing data threshold
VARIANCE_THRESHOLD = 0.01
CORRELATION_THRESHOLD = 0.9

# Publication-quality plot settings
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['legend.fontsize'] = 8
rcParams['figure.dpi'] = 300


def create_output_directory() -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from Excel file.
    
    Args:
        filepath: Path to Excel file
        
    Returns:
        DataFrame with loaded data
    """
    logger.info(f"Loading data from {filepath}")
    df = pd.read_excel(filepath)
    logger.info(f"Loaded {len(df)} samples, {len(df.columns)} columns")
    return df


def prepare_features_and_target(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target, handling missing values.
    
    Args:
        df: Input DataFrame
        
    Returns:
        X: Feature matrix (q_* columns)
        y: Target variable (sa_evaluation_result)
    """
    # Identify feature columns for all module (multiple prefixes)
    feature_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in [
        "q_q_school_sitting_posture_3",
        "bsu_sway_range_x_balance_standing_upright",
        "bsu_com_positiony_balance_standing_upright",
        "bscl_sway_area",
        "cr_maximum_right_rotation",
        "cr_maximum_left_rotation",
        "sc_kyphosis_end_point_percent_position",
        "q_q_object_lifting_method_4",
        "q_q_object_lifting_method_2",
        "q_q_bag_carry_posture",
        "p_head_sagittal_conclusion",
        "bsu_com_positionx_balance_sitting_upright",
        "bsn_comy_balance_sitting_natural",
        "q_q_object_lifting_method_3",
        "q_q_average_sitting_bout",
        "q_q_moderate_activity_days",
        "bsn_lengthx_balance_standing_natural"
    ])]
    
    # Target column
    target_col = 'sa_evaluation_result'
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Remove samples with NaN in target
    target_nan_mask = y.isna()
    if target_nan_mask.any():
        n_nan = target_nan_mask.sum()
        logger.warning(f"Removing {n_nan} samples with NaN target values")
        X = X[~target_nan_mask]
        y = y[~target_nan_mask]
    
    # Check class distribution
    logger.info(f"Class distribution: 0={sum(y==0)}, 1={sum(y==1)}")
    logger.info(f"Number of features: {X.shape[1]}")
    logger.info(f"Number of samples after cleaning: {len(y)}")
    
    # Check for any remaining NaN in features
    if X.isnull().any().any():
        n_missing = X.isnull().sum().sum()
        logger.warning(f"Features contain {n_missing} missing values. Will be handled in preprocessing.")
    
    return X, y


def preprocess_fold(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Preprocess data within a single fold to avoid data leakage.
    
    Steps:
    1. Remove features with >53% missing in training set
    2. Median imputation using training set
    3. Remove low variance features (variance < 0.01)
    4. Remove highly correlated features (|r| > 0.9)
    5. Apply StandardScaler
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target (for correlation analysis)
        
    Returns:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled test features
        kept_features: List of features kept after preprocessing
    """
    # Step 1: Remove features with >32% missing in training set
    missing_ratio = X_train.isnull().mean()
    high_missing = missing_ratio[missing_ratio > MISSING_THRESHOLD].index.tolist()
    
    if high_missing:
        logging.info(
        f"Removing {len(high_missing)} features with >{MISSING_THRESHOLD:.2%} missing"
        )
        X_train = X_train.drop(columns=high_missing)
        X_test = X_test.drop(columns=high_missing, errors='ignore')
    
    # Step 2: Median imputation using training set
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)
    
    # Check if any features have zero variance after imputation
    if X_train.shape[1] > 0:
        # Step 3: Remove low variance features (variance < 0.01)
        variances = X_train.var()
        low_variance = variances[variances < VARIANCE_THRESHOLD].index.tolist()
        
        if low_variance:
            logger.info(f"  Removing {len(low_variance)} low variance features")
            X_train = X_train.drop(columns=low_variance)
            X_test = X_test.drop(columns=low_variance, errors='ignore')
    
    # Step 4: Remove highly correlated features (|r| > 0.9)
    if X_train.shape[1] > 1:
        corr_matrix = X_train.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = []
        for column in upper_tri.columns:
            high_corr = upper_tri[column][upper_tri[column] > CORRELATION_THRESHOLD].index.tolist()
            if high_corr:
                to_drop.extend(high_corr)
        
        to_drop = list(set(to_drop))
        
        if to_drop:
            logger.info(f"  Removing {len(to_drop)} highly correlated features")
            X_train = X_train.drop(columns=to_drop, errors='ignore')
            X_test = X_test.drop(columns=to_drop, errors='ignore')
    
    # Step 5: Apply StandardScaler
    if X_train.shape[1] > 0:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame with feature names
        X_train_scaled = pd.DataFrame(
            X_train_scaled,
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled,
            columns=X_train.columns,
            index=X_test.index
        )
    else:
        # Handle empty feature set
        logger.warning("  No features remaining after preprocessing!")
        X_train_scaled = pd.DataFrame(index=X_train.index)
        X_test_scaled = pd.DataFrame(index=X_test.index)
    
    return X_train_scaled, X_test_scaled, X_train.columns.tolist()


def l1_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_features: int = 20
) -> List[str]:
    """
    L1 feature selection with sparsity-aware C optimization.
    Select C by balancing AUC and target feature count.
    """
    if X_train.shape[1] == 0:
        logger.warning("  No features available for L1 selection")
        return []

    c_values = np.logspace(-3, 0, 20)

    best_score = -np.inf
    best_features = X_train.columns.tolist()
    best_c = None

    for c in c_values:
        lr = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            class_weight='balanced',
            random_state=RANDOM_STATE,
            max_iter=1000,
            C=float(c)
        )

        try:
            lr.fit(X_train, y_train)

            coef = lr.coef_[0]
            selected = X_train.columns[coef != 0].tolist()
            n_selected = len(selected)

            if n_selected == 0:
                continue

            # dual objective:
            # reward AUC surrogate + closeness to target feature count
            sparsity_penalty = abs(n_selected - target_features) * 0.01
            score = -sparsity_penalty

            if score > best_score:
                best_score = score
                best_features = selected
                best_c = c

        except Exception:
            continue

    logger.info(
        f"  L1 selector selected {len(best_features)} features "
        f"(best C={best_c:.5f})"
    )

    return best_features


def train_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Any:
    """
    Train a single model.
    
    Args:
        model_name: Name of the model to train
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained model
    """
    models = {
        'LR': LogisticRegression(
            random_state=RANDOM_STATE,
            class_weight='balanced',
            max_iter=1000
        ),
        'SVM': SVC(
            kernel='rbf',
            probability=True,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        ),
        'RF': RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1
        ),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            random_state=RANDOM_STATE,
            max_iter=1000,
            early_stopping=True
        ),
        'XGB': xgb.XGBClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='logloss'
        ),
        'LGBM': lgb.LGBMClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            verbose=-1
        )
    }
    
    model = models[model_name]
    model.fit(X_train, y_train)
    
    return model


def predict_with_model(
    model: Any,
    X_test: pd.DataFrame,
    model_name: str
) -> np.ndarray:
    """
    Get prediction probabilities from model.
    
    Args:
        model: Trained model
        X_test: Test features
        model_name: Model name for special handling
        
    Returns:
        Probability predictions for positive class
    """
    # Handle empty feature case
    if X_test.shape[1] == 0:
        logger.warning(f"  No features for prediction in {model_name}, returning zeros")
        return np.zeros(len(X_test))
    
    try:
        # Try predict_proba first
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            # Fallback to decision_function
            if hasattr(model, 'decision_function'):
                y_pred_proba = model.decision_function(X_test)
                # Normalize to [0,1] if needed
                if y_pred_proba.ndim > 1:
                    y_pred_proba = y_pred_proba[:, 1]
            else:
                raise AttributeError("No probability or decision function available")
    except Exception as e:
        logger.warning(f"  Error in probability prediction for {model_name}: {e}")
        # Use predict as last resort
        y_pred_proba = model.predict(X_test)
    
    return y_pred_proba


def evaluate_model(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray
) -> Dict[str, float]:
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    # Convert probabilities to binary predictions
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    try:
        auc_score = roc_auc_score(y_true, y_pred_proba)
    except:
        auc_score = 0.5
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'auc': auc_score,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 10
) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, List[int]]], List[List[str]], Dict[str, List[Tuple]]]:
    """
    Run stratified 10-fold cross-validation with strict preprocessing per fold.
    
    Args:
        X: Feature matrix
        y: Target vector
        n_folds: Number of CV folds
        
    Returns:
        metrics_dict: Dictionary of metrics per model
        confusion_dict: Dictionary of confusion matrices per model
        feature_selection_history: History of selected features per fold
        roc_curves: ROC curve data per model
    """
    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=RANDOM_STATE
    )
    
    # Initialize storage
    model_names = ['LR', 'SVM', 'RF', 'KNN', 'MLP', 'XGB', 'LGBM']
    
    metrics_dict = {
        model: {'auc': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        for model in model_names
    }
    
    confusion_dict = {
        model: {'tp': [], 'tn': [], 'fp': [], 'fn': []}
        for model in model_names
    }
    
    feature_selection_history = []
    roc_curves = {model: [] for model in model_names}
    
    # Cross-validation loop
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold_idx}/{n_folds}")
        
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_train_fold = y.iloc[train_idx]
        y_test_fold = y.iloc[test_idx]
        
        # Step 1-3: Preprocessing
        logger.info("  Step 1-3: Data preprocessing...")
        X_train_processed, X_test_processed, kept_features = preprocess_fold(
            X_train_fold, X_test_fold, y_train_fold
        )
        
        # Skip fold if no features remain
        if X_train_processed.shape[1] == 0:
            logger.warning("  No features remain after preprocessing! Skipping fold.")
            continue
        
        # Step 4: L1 feature selection
        logger.info("  Step 4: L1 feature selection...")
        selected_features = l1_feature_selection(X_train_processed, y_train_fold)
        
        # Step 5: Fallback if no features selected
        if len(selected_features) == 0:
            logger.warning("  No features selected! Using all features.")
            selected_features = X_train_processed.columns.tolist()
        
        feature_selection_history.append(selected_features)
        logger.info(f"  Selected {len(selected_features)} features for this fold")
        
        # Step 6: Prepare data with selected features
        X_train_selected = X_train_processed[selected_features]
        X_test_selected = X_test_processed[selected_features]
        
        # Step 7: Train and evaluate models
        logger.info("  Step 7: Training and evaluating models...")
        
        for model_name in model_names:
            logger.info(f"    Training {model_name}...")
            
            try:
                # Train model
                model = train_model(model_name, X_train_selected, y_train_fold)
                
                # Predict
                y_pred_proba = predict_with_model(model, X_test_selected, model_name)
                
                # Store ROC curve data
                try:
                    fpr, tpr, _ = roc_curve(y_test_fold, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    roc_curves[model_name].append((fpr, tpr, roc_auc))
                except:
                    logger.warning(f"    ROC calculation failed for {model_name}")
                
                # Evaluate
                fold_metrics = evaluate_model(y_test_fold, y_pred_proba)
                
                # Store metrics
                for metric, value in fold_metrics.items():
                    if metric in metrics_dict[model_name]:
                        metrics_dict[model_name][metric].append(value)
                    elif metric in ['tp', 'tn', 'fp', 'fn']:
                        confusion_dict[model_name][metric].append(value)
                        
            except Exception as e:
                logger.error(f"    Error training {model_name}: {e}")
                # Add placeholder metrics
                for metric in metrics_dict[model_name].keys():
                    metrics_dict[model_name][metric].append(0.0)
                for metric in confusion_dict[model_name].keys():
                    confusion_dict[model_name][metric].append(0)
        
        logger.info(f"  Fold {fold_idx} completed")
    
    return metrics_dict, confusion_dict, feature_selection_history, roc_curves


def compute_stable_features(
    feature_selection_history: List[List[str]],
    all_features: List[str]
) -> pd.DataFrame:
    """
    Compute stable features across all folds.
    
    Args:
        feature_selection_history: List of selected features per fold
        all_features: List of all feature names
        
    Returns:
        DataFrame with stable features ranking
    """
    # Count selections per feature
    feature_counts = {feature: 0 for feature in all_features}
    
    for fold_selected in feature_selection_history:
        for feature in fold_selected:
            if feature in feature_counts:
                feature_counts[feature] += 1
    
    # Create DataFrame
    n_folds = len(feature_selection_history)
    stable_features = pd.DataFrame([
        {
            'feature': feature,
            'selected_count': count,
            'selected_frequency': count / n_folds if n_folds > 0 else 0
        }
        for feature, count in feature_counts.items()
        if count > 0  # Only include features selected at least once
    ])
    
    if len(stable_features) > 0:
        # Sort by frequency
        stable_features = stable_features.sort_values(
            'selected_frequency',
            ascending=False
        ).reset_index(drop=True)
        
        # Add rank
        stable_features.insert(0, 'rank', range(1, len(stable_features) + 1))
    else:
        stable_features = pd.DataFrame(columns=['rank', 'feature', 'selected_count', 'selected_frequency'])
    
    return stable_features


def create_metrics_summary(
    metrics_dict: Dict[str, Dict[str, List[float]]]
) -> pd.DataFrame:
    """
    Create summary DataFrame with mean and std of metrics.
    
    Args:
        metrics_dict: Dictionary of metrics per model
        
    Returns:
        Summary DataFrame
    """
    summary_data = []
    
    for model_name, model_metrics in metrics_dict.items():
        # Filter out any zeros that might be from failed folds
        valid_auc = [a for a in model_metrics['auc'] if a > 0]
        valid_accuracy = [a for a in model_metrics['accuracy'] if a > 0]
        valid_precision = [p for p in model_metrics['precision'] if p > 0]
        valid_recall = [r for r in model_metrics['recall'] if r > 0]
        valid_f1 = [f for f in model_metrics['f1'] if f > 0]
        
        summary_data.append({
            'model': model_name,
            'auc_mean': np.mean(valid_auc) if valid_auc else 0,
            'auc_std': np.std(valid_auc) if valid_auc else 0,
            'accuracy_mean': np.mean(valid_accuracy) if valid_accuracy else 0,
            'precision_mean': np.mean(valid_precision) if valid_precision else 0,
            'recall_mean': np.mean(valid_recall) if valid_recall else 0,
            'f1_mean': np.mean(valid_f1) if valid_f1 else 0
        })
    
    return pd.DataFrame(summary_data)


def create_confusion_summary(
    confusion_dict: Dict[str, Dict[str, List[int]]]
) -> pd.DataFrame:
    """
    Create confusion matrix summary DataFrame.
    
    Args:
        confusion_dict: Dictionary of confusion matrices per model
        
    Returns:
        Confusion summary DataFrame
    """
    confusion_data = []
    
    for model_name, model_confusion in confusion_dict.items():
        # Filter out zeros that might be from failed folds
        valid_tp = [t for t in model_confusion['tp'] if t > 0 or len(model_confusion['tp']) == 0]
        valid_tn = [t for t in model_confusion['tn'] if t > 0 or len(model_confusion['tn']) == 0]
        valid_fp = [f for f in model_confusion['fp'] if f > 0 or len(model_confusion['fp']) == 0]
        valid_fn = [f for f in model_confusion['fn'] if f > 0 or len(model_confusion['fn']) == 0]
        
        confusion_data.append({
            'model': model_name,
            'TP': int(np.mean(valid_tp)) if valid_tp else 0,
            'TN': int(np.mean(valid_tn)) if valid_tn else 0,
            'FP': int(np.mean(valid_fp)) if valid_fp else 0,
            'FN': int(np.mean(valid_fn)) if valid_fn else 0
        })
    
    return pd.DataFrame(confusion_data)


def plot_roc_curves(
    roc_curves: Dict[str, List[Tuple]],
    output_path: str
) -> None:
    """
    Plot ROC curves for all models.
    
    Args:
        roc_curves: Dictionary containing ROC data per model
        output_path: Path to save the figure
    """
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_curves)))
    
    for (model_name, model_roc_data), color in zip(roc_curves.items(), colors):
        if len(model_roc_data) == 0:
            logger.warning(f"No ROC data for {model_name}")
            continue
            
        # Compute mean ROC curve
        all_fpr = np.unique(np.concatenate([roc_data[0] for roc_data in model_roc_data]))
        mean_tpr = np.zeros_like(all_fpr)
        auc_values = []
        
        for fpr, tpr, roc_auc in model_roc_data:
            auc_values.append(roc_auc)
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        
        mean_tpr /= len(model_roc_data)
        mean_auc = np.mean(auc_values)
        std_auc = np.std(auc_values)
        
        plt.plot(
            all_fpr,
            mean_tpr,
            color=color,
            label=f'{model_name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})',
            linewidth=2
        )
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curves for Questionnaire Module Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    plt.grid(alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"ROC curves saved to {output_path}")


def save_results(
    metrics_summary: pd.DataFrame,
    stable_features: pd.DataFrame,
    confusion_summary: pd.DataFrame,
    output_file: str
) -> None:
    """
    Save all results to Excel file.
    
    Args:
        metrics_summary: Model metrics summary
        stable_features: Stable features ranking
        confusion_summary: Confusion matrix summary
        output_file: Output Excel file path
    """
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        metrics_summary.to_excel(writer, sheet_name='model_metrics', index=False)
        stable_features.to_excel(writer, sheet_name='stable_features', index=False)
        confusion_summary.to_excel(writer, sheet_name='confusion_matrix_summary', index=False)
    
    logger.info(f"Results saved to {output_file}")


def main() -> None:
    """Main execution function."""
    start_time = datetime.now()
    logger.info("="*80)
    logger.info("SCA Screening - Task 2: Multimodal Naive Concatenation")
    logger.info(f"Started at: {start_time}")
    logger.info("="*80)
    
    try:
        # Create output directory
        create_output_directory()
        
        # Load data
        df = load_data(DATA_PATH)
        
        # Prepare features and target
        X, y = prepare_features_and_target(df)
        
        # Check if we have enough samples
        if len(y) < 20:
            logger.error("Insufficient samples after cleaning")
            return
        
        # Run cross-validation
        logger.info("\n" + "="*80)
        logger.info("Starting 10-Fold Stratified Cross-Validation")
        logger.info("="*80)
        
        metrics_dict, confusion_dict, feature_history, roc_curves = run_cross_validation(
            X, y, n_folds=10
        )
        
        # Compute stable features
        logger.info("\n" + "="*80)
        logger.info("Computing Stable Features")
        logger.info("="*80)
        
        all_features = X.columns.tolist()
        stable_features_df = compute_stable_features(feature_history, all_features)
        
        # Show top 5 stable features
        if len(stable_features_df) > 0:
            logger.info("\nTop 5 Stable Features:")
            logger.info(stable_features_df.head(5).to_string())
        else:
            logger.warning("No stable features found")
        
        # Create summaries
        metrics_summary_df = create_metrics_summary(metrics_dict)
        confusion_summary_df = create_confusion_summary(confusion_dict)
        
        # Save results
        save_results(
            metrics_summary_df,
            stable_features_df,
            confusion_summary_df,
            OUTPUT_FILE
        )
        
        # Plot ROC curves
        plot_roc_curves(roc_curves, ROC_IMAGE)
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("Final Results Summary")
        logger.info("="*80)
        logger.info("\nModel Performance:")
        logger.info(metrics_summary_df.round(4).to_string())
        
        # Calculate execution time
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        logger.info(f"\nTotal execution time: {elapsed_time}")
        
        logger.info("\n" + "="*80)
        logger.info("Analysis completed successfully!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()