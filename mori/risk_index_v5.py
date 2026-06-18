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
INPUT_FILE = r"D:\科研\ais_multimodal_multitask_pipeline\data\processed\merged_all_sheets_v5.xlsx"
OUTPUT_DIR = r"D:\科研\ais_multimodal_multitask_pipeline\outputs\phase4\risk_index_v5"

TARGET_COL = "sa_evaluation_result"

MODULE_PREFIXES = [
    "bsn_", "bsu_", "bsp_", "bsec_", "bscl_", "bsb_", "bsts_", "bf_","bsn2_",'bsu2_',"p_", "bc_", "bd_", "cr_", "sc_", "qdemo_", "qpa_", "qdbp_", "qpain_", "qshp_"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# Load data with optimization
# =====================================================
print("加载数据...")
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
    """处理单个模块的函数，用于并行计算"""
    module_cols = [c for c in numeric_cols if c.startswith(prefix)]
    
    if len(module_cols) < 3:
        return None, None
    
    module_df = df[module_cols].copy()
    
    # 快速去除低方差特征
    nunique = module_df.nunique()
    high_var_cols = nunique[nunique > 2].index
    module_df = module_df[high_var_cols]
    
    if module_df.shape[1] < 3:
        return None, None
    
    # 批量处理缺失值
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(module_df)
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 使用更快的特征选择方法
    # 方法1：先用简单的相关性筛选
    correlations = np.array([np.corrcoef(X_scaled[:, i], y)[0, 1] for i in range(X_scaled.shape[1])])
    top_features = np.argsort(np.abs(correlations))[-min(20, len(correlations)):]  # 最多选20个特征
    
    if len(top_features) < 2:
        return None, None
    
    X_filtered = X_scaled[:, top_features]
    
    # 简化的L1逻辑回归（减少CV折数）
    try:
        selector = LogisticRegressionCV(
            penalty="l1",
            solver="liblinear",
            cv=3,  # 从5降到3
            scoring="roc_auc",
            max_iter=1000,  # 从2000降到1000
            random_state=42,
            n_jobs=1  # 避免嵌套并行
        )
        selector.fit(X_filtered, y)
        
        coef = selector.coef_.flatten()
        selected_mask = coef != 0
        
        if selected_mask.sum() == 0:
            # 如果没有选中的特征，使用相关性最高的特征
            selected_mask = np.zeros_like(coef, dtype=bool)
            selected_mask[np.argmax(np.abs(correlations[top_features]))] = True
            coef = np.ones_like(coef)
        
        selected_features = module_df.columns[top_features[selected_mask]]
        selected_coef = coef[selected_mask]
        
        # 计算风险分数
        module_score = np.dot(X_filtered[:, selected_mask], selected_coef)
        
        # 准备报告
        reports = []
        for feat, c in zip(selected_features, selected_coef):
            reports.append({
                "module": prefix,
                "selected_feature": feat,
                "coefficient": c
            })
        
        return module_score, reports
        
    except Exception as e:
        print(f"模块 {prefix} 处理失败: {str(e)}")
        return None, None

# 并行处理所有模块
print("开始处理模块（并行计算）...")
results = Parallel(n_jobs=-1, verbose=10)(
    delayed(process_module)(prefix, df, numeric_cols, y) 
    for prefix in MODULE_PREFIXES
)

# 收集结果
risk_dataset = pd.DataFrame(index=df.index)
module_reports = []

for module_score, reports in results:
    if module_score is not None and reports is not None:
        # 生成列名（需要确保唯一性）
        base_name = f"{reports[0]['module']}risk_score"
        col_name = base_name
        counter = 1
        while col_name in risk_dataset.columns:
            col_name = f"{base_name}_{counter}"
            counter += 1
        risk_dataset[col_name] = module_score
        module_reports.extend(reports)

# =====================================================
# 后处理：移除高度相关的风险分数
# =====================================================
print("后处理：移除冗余特征...")
if len(risk_dataset.columns) > 0:
    # 计算相关性矩阵
    corr_matrix = risk_dataset.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # 删除相关性>0.95的特征
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    risk_dataset = risk_dataset.drop(columns=to_drop)
    print(f"移除了 {len(to_drop)} 个高度相关的风险分数")

# =====================================================
# Save outputs
# =====================================================
risk_dataset[TARGET_COL] = y.values

risk_output = os.path.join(OUTPUT_DIR, "module_risk_dataset_v2.xlsx")
report_output = os.path.join(OUTPUT_DIR, "module_feature_report_v2.xlsx")

print("保存结果...")
risk_dataset.to_excel(risk_output, index=False)
pd.DataFrame(module_reports).to_excel(report_output, index=False)

print(f"模块风险数据已保存: {risk_output}")
print(f"模块特征报告已保存: {report_output}")
print(f"生成的风险分数数量: {len(risk_dataset.columns)-1}")

# =====================================================
# Quick baseline training with early stopping
# =====================================================
if len(risk_dataset) > 0 and len(risk_dataset.columns) > 1:
    print("\n开始交叉验证...")
    X_risk = risk_dataset.drop(columns=[TARGET_COL])
    
    # 如果特征太多，进一步降维
    if X_risk.shape[1] > 50:
        print(f"特征数量({X_risk.shape[1]})过多，进行PCA降维...")
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
    
    print(f"\n模块风险模型 10-fold AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
else:
    print("\n警告：没有生成有效的风险分数，跳过交叉验证")