"""
AIS Risk Prediction - v4.38 WITH KERNEL SHAP EXPLANATION (FIXED)
================================================================
基于模块风险分数的稳定模型 + Kernel SHAP黑盒解释
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
import os
import logging
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_curve, roc_auc_score, f1_score, recall_score, 
    precision_score, accuracy_score, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import shap

# 设置SHAP日志级别以减少输出
import logging as shap_logging
shap_logging.getLogger('shap').setLevel(shap_logging.WARNING)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 使用你生成的模块风险数据
DATA_PATH = r"D:\科研\ais_multimodal_multitask_pipeline\outputs\phase4\risk_index_v5.1\module_risk_dataset_v5.1.xlsx"
OUTPUT_DIR = r"D:\科研\ais_multimodal_multitask_pipeline\outputs\phase4\shapv5.1"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "final_results_with_shap.xlsx")
ROC_IMAGE = os.path.join(OUTPUT_DIR, "final_roc_curve.png")
SHAP_SUMMARY_IMAGE = os.path.join(OUTPUT_DIR, "shap_summary_plot.png")
SHAP_BAR_IMAGE = os.path.join(OUTPUT_DIR, "shap_bar_plot.png")
SHAP_BEESWARM_IMAGE = os.path.join(OUTPUT_DIR, "shap_beeswarm_plot.png")

rcParams["font.family"] = "Times New Roman"
rcParams["font.size"] = 10
rcParams["axes.labelsize"] = 12
rcParams["axes.titlesize"] = 14
rcParams["figure.dpi"] = 300


def create_output_directory():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """加载模块风险数据"""
    logger.info(f"Loading module risk data: {DATA_PATH}")
    df = pd.read_excel(DATA_PATH)
    
    # 假设列名：模块风险分数 + 目标变量
    # 请根据实际列名调整
    risk_cols = [col for col in df.columns if 'risk' in col.lower() or 'score' in col.lower()]
    
    # 找到目标列
    target_col = 'sa_evaluation_result'
    
    X = df[risk_cols].copy() if risk_cols else df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    
    # 删除缺失值
    valid_mask = y.notna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]
    
    logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
    logger.info(f"Positive: {sum(y==1)}, Negative: {sum(y==0)}")
    logger.info(f"Positive ratio: {sum(y==1)/len(y):.3f}")
    
    return X, y


def get_stable_models():
    """稳定的模型集合（增加迭代次数）"""
    return {
        "LR": LogisticRegression(
            C=1.0, 
            class_weight="balanced", 
            max_iter=2000,
            solver='lbfgs',
            random_state=RANDOM_STATE
        ),
        "RF": RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_split=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "XGB": xgb.XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            scale_pos_weight=1.5,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            use_label_encoder=False
        ),
        "LGB": lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbose=-1
        )
    }


class EnsemblePredictor:
    """封装ensemble模型用于SHAP解释"""
    def __init__(self, models, scaler):
        self.models = models
        self.scaler = scaler
    
    def predict_proba(self, X):
        """返回预测概率（用于SHAP）"""
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        X_scaled = self.scaler.transform(X_array)
        predictions = []
        for model in self.models.values():
            pred = model.predict_proba(X_scaled)[:, 1]
            predictions.append(pred)
        return np.column_stack([1 - np.mean(predictions, axis=0), np.mean(predictions, axis=0)])


def calculate_auc_ci(auc_values, confidence=0.95):
    """
    计算AUC的均值和95%置信区间
    
    Args:
        auc_values: AUC值的列表
        confidence: 置信水平（默认0.95）
    
    Returns:
        mean_auc: 平均AUC
        ci_lower: 置信区间下限
        ci_upper: 置信区间上限
    """
    if len(auc_values) == 0:
        return 0.0, 0.0, 0.0
    
    mean_auc = np.mean(auc_values)
    
    if len(auc_values) > 1:
        # 计算标准误
        std_auc = np.std(auc_values, ddof=1)
        se_auc = std_auc / np.sqrt(len(auc_values))
        
        # 使用t分布计算置信区间
        t_value = stats.t.ppf((1 + confidence) / 2, df=len(auc_values)-1)
        margin_error = t_value * se_auc
        
        ci_lower = mean_auc - margin_error
        ci_upper = mean_auc + margin_error
    else:
        ci_lower = mean_auc
        ci_upper = mean_auc
    
    return mean_auc, ci_lower, ci_upper


def evaluate_with_best_threshold(y_true, y_prob):
    """使用最佳阈值评估，包含特异性计算"""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # Youden's J statistic
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx] if len(thresholds) > best_idx else 0.5
    
    y_pred = (y_prob >= best_threshold).astype(int)
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 计算特异性
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),  # Sensitivity
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "best_threshold": best_threshold
    }


def run_stable_cv(X, y, n_folds=10):
    """稳定的交叉验证"""
    outer_cv = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=RANDOM_STATE
    )
    
    results = {
        "auc": [], 
        "accuracy": [], 
        "precision": [], 
        "recall": [], 
        "specificity": [], 
        "f1": []
    }
    all_predictions = []
    all_true = []
    roc_data = []
    
    # 存储所有训练好的模型用于SHAP解释
    all_models = []
    all_scalers = []
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Fold {fold}/{n_folds}")
        
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 训练多个模型并平均
        models = get_stable_models()
        predictions = []
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            pred = model.predict_proba(X_test_scaled)[:, 1]
            predictions.append(pred)
        
        # 存储模型和scaler
        all_models.append(models)
        all_scalers.append(scaler)
        
        # 加权平均（基于验证集性能，但这里简化）
        y_prob = np.mean(predictions, axis=0)
        
        # 评估
        metrics = evaluate_with_best_threshold(y_test, y_prob)
        
        for key in results:
            results[key].append(metrics[key])
        
        all_predictions.extend(y_prob)
        all_true.extend(y_test)
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data.append((fpr, tpr, metrics["auc"]))
        
        logger.info(f"  AUC = {metrics['auc']:.4f}, F1 = {metrics['f1']:.4f}, "
                   f"Sensitivity = {metrics['recall']:.4f}, Specificity = {metrics['specificity']:.4f}")
    
    return results, roc_data, np.array(all_true), np.array(all_predictions), all_models, all_scalers


def compute_kernel_shap(X, y, all_models, all_scalers, n_background=100):
    """
    使用Kernel SHAP对整个ensemble进行黑盒解释
    
    Args:
        X: 特征数据
        y: 目标变量
        all_models: 所有fold训练的模型列表
        all_scalers: 所有fold的scaler列表
        n_background: 背景样本数量
    
    Returns:
        shap_values: SHAP值（正类）
        X_explain: 用于解释的样本
    """
    logger.info("\n" + "=" * 80)
    logger.info("COMPUTING KERNEL SHAP VALUES FOR ENSEMBLE MODEL")
    logger.info("=" * 80)
    
    # 使用全部数据进行标准化（用于全局解释）
    global_scaler = StandardScaler()
    X_scaled_full = global_scaler.fit_transform(X)
    
    # 训练最终的ensemble模型（使用全部数据）
    final_models = get_stable_models()
    for name, model in final_models.items():
        logger.info(f"Training final {name} model on full dataset...")
        model.fit(X_scaled_full, y)
    
    # 创建ensemble预测器
    ensemble_predictor = EnsemblePredictor(final_models, global_scaler)
    
    # 选择背景数据（用于Kernel SHAP）
    if len(X) > n_background:
        logger.info(f"Selecting {n_background} background samples for Kernel SHAP...")
        # 使用分层采样确保正负样本均衡
        from sklearn.model_selection import train_test_split
        _, X_background, _, y_background = train_test_split(
            X, y, train_size=n_background, stratify=y, random_state=RANDOM_STATE
        )
    else:
        X_background = X.copy()
    
    logger.info(f"Background data shape: {X_background.shape}")
    
    # 创建Kernel SHAP解释器
    logger.info("Initializing KernelExplainer...")
    explainer = shap.KernelExplainer(
        ensemble_predictor.predict_proba, 
        X_background,
        link="logit"
    )
    
    # 选择要解释的样本
    n_explain = min(200, len(X))
    logger.info(f"Computing SHAP values for {n_explain} samples...")
    
    # 分层选择要解释的样本
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    
    n_pos_explain = min(n_explain // 2, len(pos_indices))
    n_neg_explain = min(n_explain - n_pos_explain, len(neg_indices))
    
    explain_indices = np.concatenate([
        np.random.choice(pos_indices, n_pos_explain, replace=False),
        np.random.choice(neg_indices, n_neg_explain, replace=False)
    ])
    
    X_explain = X.iloc[explain_indices]
    
    # 计算SHAP值
    logger.info("Calculating SHAP values (this may take several minutes)...")
    shap_values_raw = explainer.shap_values(X_explain, nsamples=100, silent=True)
    
    # 处理SHAP值：对于二分类问题，返回的是列表，包含两个类别的SHAP值
    if isinstance(shap_values_raw, list):
        logger.info(f"SHAP returned list of length {len(shap_values_raw)}")
        # 取正类（索引1）的SHAP值
        shap_values = shap_values_raw[1]
        logger.info(f"Selected positive class SHAP values with shape: {shap_values.shape}")
    else:
        shap_values = shap_values_raw
        logger.info(f"SHAP values shape: {shap_values.shape}")
    
    # 确保SHAP值是二维数组 (n_samples, n_features)
    if len(shap_values.shape) == 3:
        logger.warning(f"SHAP values have 3 dimensions: {shap_values.shape}, taking first channel")
        shap_values = shap_values[:, :, 0]
    elif len(shap_values.shape) > 3:
        raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")
    
    logger.info(f"Final SHAP values shape: {shap_values.shape}")
    logger.info(f"X_explain shape: {X_explain.shape}")
    
    return shap_values, X_explain


def create_shap_summary_plot(shap_values, X_explain, output_path):
    """创建SHAP summary plot（SCI论文标准格式）"""
    plt.figure(figsize=(10, 8))
    
    # 创建summary plot
    shap.summary_plot(
        shap_values, 
        X_explain,
        plot_type="dot",
        show=False,
        max_display=20,
        color_bar=True
    )
    
    # 调整图形样式
    plt.title('SHAP Feature Importance Summary', fontsize=14, fontweight='bold')
    plt.xlabel('SHAP value (impact on model output)', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"SHAP summary plot saved to: {output_path}")


def create_shap_bar_plot(shap_values, X_explain, output_path):
    """创建SHAP bar plot（特征重要性条形图）"""
    plt.figure(figsize=(10, 8))
    
    # 计算平均绝对SHAP值
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_names = X_explain.columns.tolist()
    
    # 排序
    sorted_idx = np.argsort(mean_abs_shap)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_values = mean_abs_shap[sorted_idx]
    
    # 创建水平条形图
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(sorted_features)))
    bars = plt.barh(range(len(sorted_features)), sorted_values, color=colors)
    plt.yticks(range(len(sorted_features)), sorted_features)
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, sorted_values)):
        plt.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
    
    plt.xlabel('Mean |SHAP value| (average impact on model output magnitude)', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('SHAP Feature Importance (Bar Plot)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"SHAP bar plot saved to: {output_path}")


def create_shap_beeswarm_plot(shap_values, X_explain, output_path):
    """创建SHAP beeswarm plot（蜂群图）"""
    plt.figure(figsize=(12, 8))
    
    shap.summary_plot(
        shap_values,
        X_explain,
        plot_type="dot",
        show=False,
        max_display=20,
        color_bar=True,
        alpha=0.7
    )
    
    plt.title('SHAP Value Distribution (Beeswarm Plot)', fontsize=14, fontweight='bold')
    plt.xlabel('SHAP value (impact on model output)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"SHAP beeswarm plot saved to: {output_path}")


def create_shap_dataframe(shap_values, X_explain):
    """创建SHAP值的DataFrame用于Excel输出"""
    # 计算每个特征的平均SHAP值（绝对值）
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # 计算每个特征的平均SHAP值和标准差
    mean_shap = shap_values.mean(axis=0)
    std_shap = shap_values.std(axis=0)
    
    # 计算SHAP值的95%置信区间
    n_samples = len(shap_values)
    ci_lower = mean_shap - 1.96 * std_shap / np.sqrt(n_samples)
    ci_upper = mean_shap + 1.96 * std_shap / np.sqrt(n_samples)
    
    # 创建DataFrame
    shap_df = pd.DataFrame({
        'Feature': X_explain.columns.tolist(),
        'Mean_Absolute_SHAP': mean_abs_shap,
        'Mean_SHAP': mean_shap,
        'Std_SHAP': std_shap,
        'SHAP_95CI_Lower': ci_lower,
        'SHAP_95CI_Upper': ci_upper
    })
    
    # 按重要性排序
    shap_df = shap_df.sort_values('Mean_Absolute_SHAP', ascending=False)
    shap_df['Importance_Rank'] = range(1, len(shap_df) + 1)
    shap_df['Cumulative_Importance'] = shap_df['Mean_Absolute_SHAP'].cumsum() / shap_df['Mean_Absolute_SHAP'].sum()
    
    return shap_df


def plot_final_roc(roc_data, output_path):
    """绘制最终ROC曲线（SCI论文格式）"""
    plt.figure(figsize=(8, 8))
    
    # 计算平均ROC
    all_fpr = np.unique(np.concatenate([r[0] for r in roc_data]))
    mean_tpr = np.zeros_like(all_fpr)
    aucs = []
    
    for fpr, tpr, auc_score in roc_data:
        aucs.append(auc_score)
        plt.plot(fpr, tpr, 'lightgray', alpha=0.4, linewidth=0.8)
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    
    mean_tpr /= len(roc_data)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    
    # 计算AUC的95%置信区间
    ci_lower, ci_upper = stats.t.interval(0.95, len(aucs)-1, loc=mean_auc, scale=stats.sem(aucs))
    
    # 平均ROC
    plt.plot(all_fpr, mean_tpr, 'b-', linewidth=2.5,
             label=f'Mean ROC (AUC = {mean_auc:.3f} [{ci_lower:.3f}-{ci_upper:.3f}])')
    
    # 填充95%置信区间
    tpr_std = np.zeros_like(all_fpr)
    for i, fpr_val in enumerate(all_fpr):
        tpr_vals = []
        for fpr, tpr, _ in roc_data:
            tpr_vals.append(np.interp(fpr_val, fpr, tpr))
        tpr_std[i] = np.std(tpr_vals)
    
    plt.fill_between(all_fpr, mean_tpr - 1.96*tpr_std, mean_tpr + 1.96*tpr_std,
                     alpha=0.2, color='blue', label='95% CI')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity', fontsize=12)
    plt.ylabel('Sensitivity', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10, frameon=False)
    plt.grid(alpha=0.3, linestyle='--')
    
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.2))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_final_metrics_table(results):
    """
    创建最终的指标表格，按照SCI论文要求的格式
    """
    # 计算AUC的95%置信区间
    mean_auc, ci_lower, ci_upper = calculate_auc_ci(results['auc'])
    auc_formatted = f"{mean_auc:.3f} ({ci_lower:.3f}-{ci_upper:.3f})"
    
    # 计算其他指标的均值和标准差
    metrics_summary = {}
    for metric in ['recall', 'specificity', 'precision', 'f1']:
        mean_val = np.mean(results[metric])
        std_val = np.std(results[metric])
        metrics_summary[metric] = f"{mean_val:.3f} ± {std_val:.3f}"
    
    # 创建DataFrame
    metrics_df = pd.DataFrame([{
        'Model': 'Ensemble (LR+RF+XGB+LGB)',
        'AUC (95% CI)': auc_formatted,
        'Sensitivity': metrics_summary['recall'],
        'Specificity': metrics_summary['specificity'],
        'Precision': metrics_summary['precision'],
        'F1-score': metrics_summary['f1']
    }])
    
    return metrics_df


def main():
    start = datetime.now()
    logger.info("=" * 80)
    logger.info("AIS v4.38 - WITH KERNEL SHAP EXPLANATION (FIXED)")
    logger.info("=" * 80)
    
    create_output_directory()
    
    # 加载模块风险数据
    X, y = load_data()
    
    # 运行稳定交叉验证
    results, roc_data, all_true, all_pred, all_models, all_scalers = run_stable_cv(X, y, n_folds=10)
    
    # 计算Kernel SHAP值
    shap_values, X_explain = compute_kernel_shap(
        X, y, all_models, all_scalers, n_background=100
    )
    
    # 创建SHAP数据框
    shap_df = create_shap_dataframe(shap_values, X_explain)
    
    # 创建SHAP可视化图形
    create_shap_summary_plot(shap_values, X_explain, SHAP_SUMMARY_IMAGE)
    create_shap_bar_plot(shap_values, X_explain, SHAP_BAR_IMAGE)
    create_shap_beeswarm_plot(shap_values, X_explain, SHAP_BEESWARM_IMAGE)
    
    # 计算总体指标
    overall_metrics = evaluate_with_best_threshold(all_true, all_pred)
    
    # 创建最终的指标表格
    final_metrics_df = create_final_metrics_table(results)
    
    # 计算AUC的95% CI
    mean_auc, ci_lower, ci_upper = calculate_auc_ci(results['auc'])
    
    # 创建详细的fold结果表
    detail_df = pd.DataFrame({
        'Fold': range(1, 11),
        'AUC': results['auc'],
        'Sensitivity': results['recall'],
        'Specificity': results['specificity'],
        'Precision': results['precision'],
        'F1-score': results['f1']
    })
    
    # 计算统计汇总
    stats_df = pd.DataFrame({
        'Metric': ['AUC', 'Sensitivity', 'Specificity', 'Precision', 'F1-score'],
        'Mean': [
            np.mean(results['auc']),
            np.mean(results['recall']),
            np.mean(results['specificity']),
            np.mean(results['precision']),
            np.mean(results['f1'])
        ],
        'Std': [
            np.std(results['auc']),
            np.std(results['recall']),
            np.std(results['specificity']),
            np.std(results['precision']),
            np.std(results['f1'])
        ],
        'Min': [
            np.min(results['auc']),
            np.min(results['recall']),
            np.min(results['specificity']),
            np.min(results['precision']),
            np.min(results['f1'])
        ],
        'Max': [
            np.max(results['auc']),
            np.max(results['recall']),
            np.max(results['specificity']),
            np.max(results['precision']),
            np.max(results['f1'])
        ]
    })
    
    # 保存结果到Excel
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        final_metrics_df.to_excel(writer, sheet_name='Table1_Model_Performance', index=False)
        detail_df.to_excel(writer, sheet_name='Table2_CV_Details', index=False)
        stats_df.to_excel(writer, sheet_name='Table3_Statistics', index=False)
        shap_df.to_excel(writer, sheet_name='Table4_SHAP_Importance', index=False)
        
        overall_data = [
            {'Metric': 'AUC', 'Value': overall_metrics['auc'], 'Best_Threshold': overall_metrics['best_threshold']},
            {'Metric': 'Sensitivity', 'Value': overall_metrics['recall'], 'Best_Threshold': ''},
            {'Metric': 'Specificity', 'Value': overall_metrics['specificity'], 'Best_Threshold': ''},
            {'Metric': 'Precision', 'Value': overall_metrics['precision'], 'Best_Threshold': ''},
            {'Metric': 'F1-score', 'Value': overall_metrics['f1'], 'Best_Threshold': ''},
            {'Metric': 'Accuracy', 'Value': overall_metrics['accuracy'], 'Best_Threshold': ''}
        ]
        overall_df = pd.DataFrame(overall_data)
        overall_df.to_excel(writer, sheet_name='Table5_Overall_Performance', index=False)
        
        # 保存原始SHAP值矩阵
        shap_matrix_df = pd.DataFrame(
            shap_values,
            columns=[f"{col}_SHAP" for col in X_explain.columns]
        )
        shap_matrix_df.to_excel(writer, sheet_name='Supplementary_SHAP_Matrix', index=False)
    
    # 绘制ROC
    plot_final_roc(roc_data, ROC_IMAGE)
    
    # 打印结果
    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS:")
    logger.info("=" * 80)
    logger.info("\nTable 1. Model Performance Metrics")
    logger.info("-" * 60)
    logger.info(final_metrics_df.to_string(index=False))
    
    logger.info("\nTable 4. Top 10 SHAP Feature Importance")
    logger.info("-" * 60)
    logger.info(shap_df.head(10)[['Feature', 'Mean_Absolute_SHAP', 'Mean_SHAP', 'Importance_Rank']].to_string(index=False))
    
    logger.info(f"\n{'='*80}")
    logger.info("PERFORMANCE SUMMARY:")
    logger.info(f"{'='*80}")
    logger.info(f"  AUC (95% CI):     {mean_auc:.3f} ({ci_lower:.3f}-{ci_upper:.3f})")
    logger.info(f"  AUC Range:        [{np.min(results['auc']):.4f}, {np.max(results['auc']):.4f}]")
    logger.info(f"  Mean Sensitivity: {np.mean(results['recall']):.4f} ± {np.std(results['recall']):.4f}")
    logger.info(f"  Mean Specificity: {np.mean(results['specificity']):.4f} ± {np.std(results['specificity']):.4f}")
    logger.info(f"  Mean Precision:   {np.mean(results['precision']):.4f} ± {np.std(results['precision']):.4f}")
    logger.info(f"  Mean F1-score:    {np.mean(results['f1']):.4f} ± {np.std(results['f1']):.4f}")
    logger.info(f"  Best Threshold:   {overall_metrics['best_threshold']:.4f}")
    
    if mean_auc >= 0.78:
        logger.info(f"\n✅ SUCCESS: Target AUC 0.78 achieved!")
        logger.info(f"   Final AUC = {mean_auc:.3f} ({ci_lower:.3f}-{ci_upper:.3f})")
    else:
        logger.info(f"\n❌ Target not achieved. Current AUC: {mean_auc:.3f}")
    
    logger.info(f"\nFinished in: {datetime.now() - start}")
    logger.info(f"\nResults saved to: {OUTPUT_FILE}")
    logger.info(f"ROC curve saved to: {ROC_IMAGE}")
    logger.info(f"SHAP summary plot saved to: {SHAP_SUMMARY_IMAGE}")
    logger.info(f"SHAP bar plot saved to: {SHAP_BAR_IMAGE}")
    logger.info(f"SHAP beeswarm plot saved to: {SHAP_BEESWARM_IMAGE}")


if __name__ == "__main__":
    main()