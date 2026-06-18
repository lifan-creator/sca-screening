"""
SCA Screening - Preprocessing: Baseline Statistical Analysis
================================================================================
Performs descriptive statistics and ordinal logistic regression on the merged
dataset to characterise differences between the normal and risk groups.

Pipeline:
  1. Descriptive statistics (continuous: median/IQR with Mann-Whitney U or t-test;
     categorical: n/% with chi-square or Fisher exact test)
  2. SCI-standard three-line Table 1 (baseline characteristics)
  3. Ordinal logistic regression on significant univariate variables → Table 2
  4. Publication-quality figures (gender distribution bar chart, forest plot)

Outputs are saved under the specified output directory in subdirectories
'tables/' and 'figures/'.

Author: Li Fan
Date: 2026-04-01
Version: 4.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.preprocessing import LabelEncoder
import warnings
import os
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
from openpyxl.utils import get_column_letter
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalysis:
    def __init__(self, file_path, output_dir=None):
        """Initialise the analysis object: load data and run preprocessing."""
        self.file_path = file_path
        self.df = pd.read_excel(file_path)
        self.results = {}
        self.significant_vars = []
        
        # 设置输出目录
        if output_dir:
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)
        else:
            self.output_dir = os.getcwd()
        
        # 创建输出子目录
        self.tables_dir = os.path.join(self.output_dir, 'tables')
        self.figures_dir = os.path.join(self.output_dir, 'figures')
        os.makedirs(self.tables_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # 预处理数据
        self._preprocess_data()
        
    def _preprocess_data(self):
        """Preprocess loaded data: drop unlabelled rows, map gender, classify variables."""
        print("正在进行数据预处理...")
        
        # 删除sa_evaluation_result为NaN的行
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=['sa_evaluation_result'])
        print(f"删除sa_evaluation_result为NaN的行: {initial_rows - len(self.df)}行")
        print(f"当前样本量: {len(self.df)}")
        
        # 统一处理性别变量 - 使用p_gender作为主要性别变量
        self.gender_var = None
        if 'p_gender' in self.df.columns:
            # p_gender: 1=男, 2=女
            self.df['p_gender_label'] = self.df['p_gender'].map({1: '男', 2: '女'})
            self.gender_var = 'p_gender'
            print("使用p_gender作为性别变量 (1=男, 2=女)")
        elif 'q_sex' in self.df.columns:
            # 如果没有p_gender，使用q_sex
            self.df['p_gender_label'] = self.df['q_sex'].map({1: '女', 2: '男'})
            self.gender_var = 'q_sex'
            print("使用q_sex作为性别变量")
        
        # 获取所有变量 - 排除sa_开头的和student_id
        self.all_vars = [col for col in self.df.columns 
                        if not col.startswith('sa_') and col != 'student_id']
        
        print(f"识别到变量总数: {len(self.all_vars)}个")
        
        # 批量计算缺失值
        self.missing_summary = {}
        missing_counts = self.df[self.all_vars].isna().sum()
        self.missing_summary = missing_counts[missing_counts > 0].to_dict()
        
        if self.missing_summary:
            print(f"发现 {len(self.missing_summary)} 个变量存在缺失值")
        
        # 创建分组
        self.df['group'] = self.df['sa_evaluation_result'].map({0: '正常组', 1: '风险组'})
        
        # 定义变量类型
        self._define_variable_types()
        
    def _define_variable_types(self):
        """Classify all columns as categorical or continuous based on domain knowledge and cardinality."""
        # 手动指定的关键分类变量
        categorical_vars = [
            'p_gender', 'q_sex', 'q_q_low_back_pain', 'bc_body_fat_percentage_results',
            'q_q_school_sitting_posture', 'q_q_device_reading_posture',
            'q_q_bag_style', 'q_q_bag_carry_posture', 'q_q_sleep_posture',
            'p_shoulder_low_conclusion', 'p_pelvic_tilt_conclusion',
            'p_pelvic_rotation_conclusion', 'p_head_sagittal_conclusion'
        ]
        
        # 手动指定的关键连续变量
        continuous_vars = [
            'q_age', 'bd_bmi', 'q_q_sleep_duration', 'q_q_screen_time_daily',
            'q_q_moderate_activity_days', 'q_q_vigorous_activity_days',
            'cr_flexion_extension_difference', 'cr_left_right_flexion_difference',
            'bc_left_upper_limb_muscle_mass', 'bc_right_upper_limb_muscle_mass',
            'bc_left_lower_limb_muscle_mass', 'bc_right_lower_limb_muscle_mass',
            'bd_zscore', 'bd_bqi', 'bsn_sway_area', 'bsn_energy_consumption', 'bsn_velocity_com'
        ]
        
        # 确保变量存在于数据中
        self.categorical_vars = [v for v in categorical_vars if v in self.df.columns]
        self.continuous_vars = [v for v in continuous_vars if v in self.df.columns]
        
        # 自动判断剩余变量类型（优化：使用nunique阈值）
        existing_cat_cont = set(self.categorical_vars + self.continuous_vars)
        remaining_vars = [v for v in self.all_vars if v not in existing_cat_cont]
        
        for var in remaining_vars:
            if var in self.df.columns:
                unique_count = self.df[var].nunique()
                if unique_count <= 10:
                    self.categorical_vars.append(var)
                elif pd.api.types.is_numeric_dtype(self.df[var]):
                    self.continuous_vars.append(var)
                else:
                    self.categorical_vars.append(var)
        
        print(f"识别分类变量: {len(self.categorical_vars)}个")
        print(f"识别连续变量: {len(self.continuous_vars)}个")
    
    def descriptive_statistics(self):
        """Run descriptive statistics for all variables and generate SCI-standard Table 1."""
        print("\n正在进行描述性统计分析...")
        
        # 基本统计量
        self.total_n = len(self.df)
        self.normal_n = len(self.df[self.df['sa_evaluation_result'] == 0])
        self.risk_n = len(self.df[self.df['sa_evaluation_result'] == 1])
        
        print(f"总样本量: {self.total_n} (正常组: {self.normal_n}, 风险组: {self.risk_n})")
        
        # 存储结果用于三线表
        self.baseline_table_data = []
        
        # 分析所有变量
        for var in self.all_vars:
            if var not in self.df.columns:
                continue
                
            if var in self.categorical_vars:
                self._analyze_categorical_variable(var)
            elif var in self.continuous_vars:
                self._analyze_continuous_variable(var)
        
        # 生成SCI标准三线表
        self._create_sci_baseline_table()
        
        return self.baseline_table_data
    
    def _analyze_continuous_variable(self, var):
        """Compare a continuous variable between groups using t-test or Mann-Whitney U."""
        try:
            normal_data = self.df[self.df['sa_evaluation_result'] == 0][var].dropna()
            risk_data = self.df[self.df['sa_evaluation_result'] == 1][var].dropna()
            total_data = pd.concat([normal_data, risk_data])
            
            if len(normal_data) < 3 or len(risk_data) < 3:
                return
            
            # 正态性检验
            _, normal_p = stats.shapiro(normal_data) if len(normal_data) < 5000 else (0, 0)
            _, risk_p = stats.shapiro(risk_data) if len(risk_data) < 5000 else (0, 0)
            
            # 选择检验方法
            if len(normal_data) < 5000 and normal_p > 0.05 and risk_p > 0.05:
                # 正态分布，使用t检验
                stat, p_value = stats.ttest_ind(normal_data, risk_data)
                method = "独立样本t检验"
            else:
                # 非正态分布，使用Mann-Whitney U检验
                stat, p_value = stats.mannwhitneyu(normal_data, risk_data, alternative='two-sided')
                method = "Mann-Whitney U检验"
            
            # 计算描述性统计
            normal_median = normal_data.median()
            normal_q1 = normal_data.quantile(0.25)
            normal_q3 = normal_data.quantile(0.75)
            
            risk_median = risk_data.median()
            risk_q1 = risk_data.quantile(0.25)
            risk_q3 = risk_data.quantile(0.75)
            
            # 存储结果
            result = {
                '变量名': var,
                '类型': '连续变量',
                '总体(N={})'.format(self.total_n): f"{total_data.median():.2f} ({total_data.quantile(0.25):.2f}, {total_data.quantile(0.75):.2f})",
                '正常组(N={})'.format(self.normal_n): f"{normal_median:.2f} ({normal_q1:.2f}, {normal_q3:.2f})",
                '风险组(N={})'.format(self.risk_n): f"{risk_median:.2f} ({risk_q1:.2f}, {risk_q3:.2f})",
                '统计量': round(stat, 3) if not np.isnan(stat) else 'N/A',
                'P值': round(p_value, 4) if not np.isnan(p_value) else 'N/A',
                '检验方法': method
            }
            
            self.baseline_table_data.append(result)
            
            if p_value < 0.05:
                self.significant_vars.append(var)
                
        except Exception as e:
            print(f"分析连续变量 {var} 时出错: {str(e)}")
    
    def _analyze_categorical_variable(self, var):
        """Compare a categorical variable between groups using Fisher exact test or chi-square."""
        try:
            valid_data = self.df[[var, 'sa_evaluation_result']].dropna()
            if len(valid_data) == 0:
                return
            
            # 创建列联表
            contingency_table = pd.crosstab(valid_data[var], valid_data['sa_evaluation_result'])
            
            if contingency_table.shape[0] < 2:
                return
            
            # 统计检验
            try:
                if contingency_table.shape[0] == 2 and contingency_table.shape[1] == 2:
                    # 2x2表使用Fisher精确检验
                    odds_ratio, p_value = stats.fisher_exact(contingency_table)
                    method = "Fisher精确检验"
                    stat = odds_ratio
                else:
                    # 其他情况使用卡方检验
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    method = "卡方检验"
                    stat = chi2
            except:
                method = "无法检验"
                stat = np.nan
                p_value = np.nan
            
            # 计算百分比（按列计算）
            normal_pcts = (contingency_table[0] / self.normal_n * 100).round(1)
            risk_pcts = (contingency_table[1] / self.risk_n * 100).round(1)
            
            # 格式化输出 - 详细展示每个类别
            for category in contingency_table.index:
                n_count = int(contingency_table.loc[category, 0]) if 0 in contingency_table.columns else 0
                r_count = int(contingency_table.loc[category, 1]) if 1 in contingency_table.columns else 0
                n_pct = float(normal_pcts.get(category, 0))
                r_pct = float(risk_pcts.get(category, 0))
                
                result = {
                    '变量名': f"{var} - {category}",
                    '类型': '分类变量',
                    '总体(N={})'.format(self.total_n): f"{n_count + r_count} ({(n_count + r_count)/self.total_n*100:.1f}%)",
                    '正常组(N={})'.format(self.normal_n): f"{n_count} ({n_pct}%)",
                    '风险组(N={})'.format(self.risk_n): f"{r_count} ({r_pct}%)",
                    '统计量': round(stat, 3) if not np.isnan(stat) else 'N/A',
                    'P值': round(p_value, 4) if not np.isnan(p_value) else 'N/A',
                    '检验方法': method
                }
                
                self.baseline_table_data.append(result)
            
            if p_value < 0.05:
                self.significant_vars.append(var)
                
        except Exception as e:
            print(f"分析分类变量 {var} 时出错: {str(e)}")
    
    def _create_sci_baseline_table(self):
        """Generate SCI-standard three-line Table 1 (baseline characteristics) as an Excel workbook."""
        print("\n正在生成SCI标准三线表...")
        
        if not self.baseline_table_data:
            print("没有数据可生成表格")
            return
        
        # 创建DataFrame
        table_df = pd.DataFrame(self.baseline_table_data)
        
        # 创建Excel工作簿
        wb = Workbook()
        ws = wb.active
        ws.title = "Table1_基线特征"
        
        # 设置列宽
        column_widths = {
            'A': 35,  # 变量名
            'B': 15,  # 总体
            'C': 18,  # 正常组
            'D': 18,  # 风险组
            'E': 12,  # 统计量
            'F': 12,  # P值
            'G': 20   # 检验方法
        }
        
        for col, width in column_widths.items():
            ws.column_dimensions[col].width = width
        
        # 定义样式
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        top_border = Border(
            top=Side(style='medium'),
            bottom=Side(style='thin'),
            left=Side(style='thin'),
            right=Side(style='thin')
        )
        
        bottom_border = Border(
            bottom=Side(style='medium'),
            left=Side(style='thin'),
            right=Side(style='thin')
        )
        
        header_font = Font(name='Times New Roman', size=10, bold=True)
        body_font = Font(name='Times New Roman', size=10)
        title_font = Font(name='Times New Roman', size=12, bold=True)
        
        # 添加标题
        title_row = 1
        ws.merge_cells(f'A{title_row}:G{title_row}')
        ws[f'A{title_row}'] = 'Table 1. Baseline characteristics of study participants'
        ws[f'A{title_row}'].font = title_font
        ws[f'A{title_row}'].alignment = Alignment(horizontal='center', vertical='center')
        ws.row_dimensions[title_row].height = 30
        
        # 添加表头
        header_row = 3
        headers = ['Characteristic', f'Total\n(N={self.total_n})', 
                  f'Normal group\n(N={self.normal_n})', f'Risk group\n(N={self.risk_n})',
                  'Statistic', 'P value', 'Test method']
        
        for col_idx, header in enumerate(headers, 1):
            cell = ws.cell(row=header_row, column=col_idx, value=header)
            cell.font = header_font
            cell.border = top_border  # 顶线使用粗线
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        ws.row_dimensions[header_row].height = 40
        
        # 添加数据行
        data_start_row = header_row + 1
        current_row = data_start_row
        
        # 按变量分组显示
        displayed_vars = set()
        
        for idx, row in table_df.iterrows():
            var_name = row['变量名']
            base_var = var_name.split(' - ')[0] if ' - ' in var_name else var_name
            
            # 如果是变量的第一个类别，添加变量分组标题
            if base_var not in displayed_vars and ' - ' in var_name:
                # 添加变量分组标题行
                ws.merge_cells(f'A{current_row}:G{current_row}')
                cell = ws.cell(row=current_row, column=1, value=base_var)
                cell.font = Font(name='Times New Roman', size=10, bold=True, italic=True)
                cell.border = thin_border
                cell.alignment = Alignment(horizontal='left', vertical='center')
                ws.row_dimensions[current_row].height = 22
                current_row += 1
                displayed_vars.add(base_var)
            
            # 添加数据行
            for col_idx, col_name in enumerate(['变量名', f'总体(N={self.total_n})', 
                                               f'正常组(N={self.normal_n})', f'风险组(N={self.risk_n})',
                                               '统计量', 'P值', '检验方法'], 1):
                cell = ws.cell(row=current_row, column=col_idx, value=row[col_name])
                cell.font = body_font
                cell.border = thin_border
                cell.alignment = Alignment(horizontal='center', vertical='center')
                
                # P值加粗显示（如果显著）
                if col_name == 'P值' and isinstance(row['P值'], (int, float)) and row['P值'] < 0.05:
                    cell.font = Font(name='Times New Roman', size=10, bold=True, color='FF0000')
            
            ws.row_dimensions[current_row].height = 20
            current_row += 1
        
        # 添加底线（粗线）
        last_data_row = current_row - 1
        for col_idx in range(1, 8):
            cell = ws.cell(row=last_data_row, column=col_idx)
            current_border = cell.border
            cell.border = Border(
                top=current_border.top,
                bottom=Side(style='medium'),  # 底线使用粗线
                left=current_border.left,
                right=current_border.right
            )
        
        # 添加表注
        note_row = current_row + 1
        ws.merge_cells(f'A{note_row}:G{note_row}')
        ws[f'A{note_row}'] = 'Note: Data are presented as median (IQR) for continuous variables and n (%) for categorical variables.'
        ws[f'A{note_row}'].font = Font(name='Times New Roman', size=9, italic=True)
        ws[f'A{note_row}'].alignment = Alignment(horizontal='left', vertical='center')
        
        # 添加P值说明
        note_row2 = note_row + 1
        ws.merge_cells(f'A{note_row2}:G{note_row2}')
        ws[f'A{note_row2}'] = 'P values in bold indicate statistical significance (P < 0.05).'
        ws[f'A{note_row2}'].font = Font(name='Times New Roman', size=9, italic=True)
        ws[f'A{note_row2}'].alignment = Alignment(horizontal='left', vertical='center')
        
        # 保存文件
        table_path = os.path.join(self.tables_dir, 'Table1_Baseline_Characteristics.xlsx')
        wb.save(table_path)
        print(f"SCI标准三线表已保存: {table_path}")
        
        self.baseline_df = table_df
    
    def ordinal_logistic_regression(self):
        """Fit ordinal logistic regression on significant univariate variables and export Table 2."""
        print("\n正在进行有序逻辑回归分析...")
        
        if len(self.significant_vars) == 0:
            print("没有显著变量可纳入回归模型")
            return None
        
        # 筛选变量（缺失率<20%且不是标签变量）
        good_vars = []
        for var in self.significant_vars:
            if var in self.df.columns and var not in ['p_gender_label', 'q_sex_label']:
                missing_rate = self.df[var].isna().sum() / len(self.df)
                if missing_rate < 0.2:
                    good_vars.append(var)
        
        if len(good_vars) == 0:
            print("没有合适的变量纳入回归模型")
            return None
        
        print(f"筛选出 {len(good_vars)} 个变量纳入回归模型")
        
        # 准备数据
        model_df = self.df[good_vars + ['sa_evaluation_result']].dropna()
        print(f"完整数据样本量: {len(model_df)}")
        
        if len(model_df) < 50:
            print("警告：样本量较小，回归结果可能不稳定")
            if len(model_df) < 30:
                return None
        
        X = model_df[good_vars].copy()
        y = model_df['sa_evaluation_result'].astype(int)
        
        # 处理变量
        X_processed = pd.DataFrame()
        for var in X.columns:
            if var in self.categorical_vars:
                # 标签编码
                le = LabelEncoder()
                X_processed[var] = le.fit_transform(X[var].astype(str))
            else:
                # 标准化连续变量
                X_processed[var] = (X[var] - X[var].mean()) / X[var].std()
        
        # 确保数据是数值类型
        X_processed = X_processed.astype(float)
        
        try:
            # 拟合有序逻辑回归
            model = OrderedModel(y, X_processed, distr='logit')
            result = model.fit(method='bfgs', disp=False, maxiter=200)
            
            # 提取结果并计算OR值
            or_results = []
            params = result.params
            conf_int = result.conf_int()
            p_values = result.pvalues
            
            for param in params.index:
                if not param.startswith('x'):
                    continue
                
                var_name = param[1:]  # 去掉'x'前缀
                coef = float(params[param])
                or_value = float(np.exp(coef))
                ci_lower = float(np.exp(conf_int.loc[param, 0]))
                ci_upper = float(np.exp(conf_int.loc[param, 1]))
                p_val = float(p_values[param])
                
                or_results.append({
                    '变量': var_name,
                    'β系数': round(coef, 4),
                    '标准误': round(float(result.bse[param]), 4),
                    'Wald χ²': round((coef / float(result.bse[param]))**2, 2),
                    'OR值': round(or_value, 3),
                    '95% CI 下限': round(ci_lower, 3),
                    '95% CI 上限': round(ci_upper, 3),
                    'P值': round(p_val, 4)
                })
            
            self.ordinal_results = pd.DataFrame(or_results)
            
            # 模型拟合信息
            self.model_fit = {
                '伪R²': round(float(result.prsquared), 4),
                '对数似然值': round(float(result.llf), 2),
                'AIC': round(float(result.aic), 2),
                'BIC': round(float(result.bic), 2)
            }
            
            # 生成回归结果三线表
            self._create_regression_table()
            
            print("有序逻辑回归完成")
            
        except Exception as e:
            print(f"有序逻辑回归失败: {e}")
            self.ordinal_results = None
            self.model_fit = None
        
        return self.ordinal_results
    
    def _create_regression_table(self):
        """Generate SCI-standard three-line Table 2 (ordinal logistic regression results) as an Excel workbook."""
        print("\n正在生成回归分析三线表...")
        
        if self.ordinal_results is None or len(self.ordinal_results) == 0:
            return
        
        wb = Workbook()
        ws = wb.active
        ws.title = "Table2_Regression"
        
        # 设置列宽
        column_widths = {
            'A': 30, 'B': 10, 'C': 10, 'D': 10,
            'E': 12, 'F': 12, 'G': 12, 'H': 10
        }
        
        for col, width in column_widths.items():
            ws.column_dimensions[col].width = width
        
        # 定义样式
        thin_border = Border(
            left=Side(style='thin'), right=Side(style='thin'),
            top=Side(style='thin'), bottom=Side(style='thin')
        )
        
        top_border = Border(
            top=Side(style='medium'), bottom=Side(style='thin'),
            left=Side(style='thin'), right=Side(style='thin')
        )
        
        header_font = Font(name='Times New Roman', size=10, bold=True)
        body_font = Font(name='Times New Roman', size=10)
        title_font = Font(name='Times New Roman', size=12, bold=True)
        
        # 添加标题
        title_row = 1
        ws.merge_cells(f'A{title_row}:H{title_row}')
        ws[f'A{title_row}'] = 'Table 2. Ordinal logistic regression analysis of factors associated with scoliosis risk'
        ws[f'A{title_row}'].font = title_font
        ws[f'A{title_row}'].alignment = Alignment(horizontal='center', vertical='center')
        ws.row_dimensions[title_row].height = 30
        
        # 添加表头
        header_row = 3
        headers = ['Variable', 'β', 'SE', 'Wald χ²', 'OR', '95% CI Lower', '95% CI Upper', 'P value']
        
        for col_idx, header in enumerate(headers, 1):
            cell = ws.cell(row=header_row, column=col_idx, value=header)
            cell.font = header_font
            cell.border = top_border
            cell.alignment = Alignment(horizontal='center', vertical='center')
        ws.row_dimensions[header_row].height = 25
        
        # 添加数据
        data_start_row = header_row + 1
        for row_idx, (_, data_row) in enumerate(self.ordinal_results.iterrows()):
            current_row = data_start_row + row_idx
            
            # 写入数据
            ws.cell(row=current_row, column=1, value=data_row['变量']).font = body_font
            ws.cell(row=current_row, column=2, value=data_row['β系数']).font = body_font
            ws.cell(row=current_row, column=3, value=data_row['标准误']).font = body_font
            ws.cell(row=current_row, column=4, value=data_row['Wald χ²']).font = body_font
            ws.cell(row=current_row, column=5, value=data_row['OR值']).font = body_font
            ws.cell(row=current_row, column=6, value=data_row['95% CI 下限']).font = body_font
            ws.cell(row=current_row, column=7, value=data_row['95% CI 上限']).font = body_font
            
            # P值特殊处理
            p_cell = ws.cell(row=current_row, column=8, value=data_row['P值'])
            if data_row['P值'] < 0.05:
                p_cell.font = Font(name='Times New Roman', size=10, bold=True, color='FF0000')
            else:
                p_cell.font = body_font
            
            # 添加边框
            for col in range(1, 9):
                ws.cell(row=current_row, column=col).border = thin_border
                ws.cell(row=current_row, column=col).alignment = Alignment(horizontal='center', vertical='center')
            
            ws.row_dimensions[current_row].height = 20
        
        # 添加底线（粗线）
        last_data_row = data_start_row + len(self.ordinal_results) - 1
        for col_idx in range(1, 9):
            cell = ws.cell(row=last_data_row, column=col_idx)
            cell.border = Border(
                top=Side(style='thin'),
                bottom=Side(style='medium'),
                left=Side(style='thin'),
                right=Side(style='thin')
            )
        
        # 添加模型拟合信息
        fit_start_row = last_data_row + 2
        if self.model_fit:
            fit_info = [
                f"Model fit: Pseudo R² = {self.model_fit['伪R²']}",
                f"Log-likelihood = {self.model_fit['对数似然值']}",
                f"AIC = {self.model_fit['AIC']}, BIC = {self.model_fit['BIC']}"
            ]
            
            for i, info in enumerate(fit_info):
                ws.merge_cells(f'A{fit_start_row + i}:H{fit_start_row + i}')
                ws[f'A{fit_start_row + i}'] = info
                ws[f'A{fit_start_row + i}'].font = Font(name='Times New Roman', size=9, italic=True)
        
        # 添加表注
        note_row = fit_start_row + len(fit_info) + 1 if self.model_fit else fit_start_row
        ws.merge_cells(f'A{note_row}:H{note_row}')
        ws[f'A{note_row}'] = 'Note: OR, odds ratio; CI, confidence interval; SE, standard error.'
        ws[f'A{note_row}'].font = Font(name='Times New Roman', size=9, italic=True)
        
        note_row2 = note_row + 1
        ws.merge_cells(f'A{note_row2}:H{note_row2}')
        ws[f'A{note_row2}'] = 'P values in bold indicate statistical significance (P < 0.05).'
        ws[f'A{note_row2}'].font = Font(name='Times New Roman', size=9, italic=True)
        
        # 保存
        table_path = os.path.join(self.tables_dir, 'Table2_Ordinal_Logistic_Regression.xlsx')
        wb.save(table_path)
        print(f"回归分析三线表已保存: {table_path}")
    
    def create_visualizations(self):
        """Create publication-quality figures (gender distribution bar chart and forest plot)."""
        print("\n正在创建可视化图表...")
        
        # 设置matplotlib参数以符合SCI要求
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
        
        normal_group = self.df[self.df['sa_evaluation_result'] == 0]
        risk_group = self.df[self.df['sa_evaluation_result'] == 1]
        
        # 创建关键图表
        if self.gender_var and 'p_gender_label' in self.df.columns:
            self._create_gender_figure(normal_group, risk_group)
        
        if self.ordinal_results is not None and len(self.ordinal_results) > 0:
            self._create_forest_plot_sci()
    
    def _create_gender_figure(self, normal_group, risk_group):
        """Generate SCI-standard gender distribution bar chart (Figure 1)."""
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # 统计性别分布
        normal_gender = normal_group['p_gender_label'].value_counts()
        risk_gender = risk_group['p_gender_label'].value_counts()
        
        # 确保顺序：男、女
        categories = ['男', '女'] if '男' in normal_gender.index else normal_gender.index.tolist()
        
        gender_data = {
            'Normal': [normal_gender.get(cat, 0) for cat in categories],
            'Risk': [risk_gender.get(cat, 0) for cat in categories]
        }
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, gender_data['Normal'], width, 
                      label='Normal group', color='#4472C4', edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, gender_data['Risk'], width,
                      label='Risk group', color='#ED7D31', edgecolor='black', linewidth=0.5)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Gender', fontsize=11)
        ax.set_ylabel('Number of participants', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        
        # 移除顶部和右侧边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'Figure1_Gender_Distribution.tiff'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.figures_dir, 'Figure1_Gender_Distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("性别分布图已保存")
    
    def _create_forest_plot_sci(self):
        """Generate SCI-standard forest plot of OR values with 95% CI (Figure 2)."""
        fig, ax = plt.subplots(figsize=(10, max(6, len(self.ordinal_results) * 0.4)))
        
        or_df = self.ordinal_results.sort_values('OR值', ascending=True)
        
        y_pos = np.arange(len(or_df))
        
        # 绘制OR值和95%CI
        for i, (_, row) in enumerate(or_df.iterrows()):
            or_val = row['OR值']
            ci_low = row['95% CI 下限']
            ci_high = row['95% CI 上限']
            
            # 根据显著性选择颜色
            color = '#D62728' if row['P值'] < 0.05 else '#1F77B4'
            
            # 绘制误差线
            ax.errorbar(or_val, i, 
                       xerr=[[or_val - ci_low], [ci_high - or_val]],
                       fmt='o', color=color, capsize=3, markersize=8,
                       markeredgecolor='black', markeredgewidth=0.5)
        
        # 添加参考线
        ax.axvline(x=1, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(or_df['变量'], fontsize=10)
        ax.set_xlabel('Odds Ratio (95% CI)', fontsize=11)
        ax.invert_yaxis()
        
        # 移除顶部和右侧边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'Figure2_Forest_Plot.tiff'),
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.figures_dir, 'Figure2_Forest_Plot.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("森林图已保存")
    
    def export_results(self, output_filename='Statistical_Analysis_Results.xlsx'):
        """Export analysis overview, variable list, and regression results to an Excel workbook."""
        print("\n正在导出分析结果...")
        
        output_path = os.path.join(self.output_dir, output_filename)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 数据概览
            overview = {
                'Total participants': self.total_n,
                'Normal group': self.normal_n,
                'Risk group': self.risk_n,
                'Categorical variables': len(self.categorical_vars),
                'Continuous variables': len(self.continuous_vars),
                'Significant variables (P<0.05)': len(self.significant_vars)
            }
            pd.DataFrame([overview]).to_excel(writer, sheet_name='Overview', index=False)
            
            # 变量列表
            var_df = pd.DataFrame({
                'Variable': self.all_vars,
                'Type': ['Categorical' if v in self.categorical_vars else 'Continuous' for v in self.all_vars],
                'Significant': ['Yes' if v in self.significant_vars else 'No' for v in self.all_vars]
            })
            var_df.to_excel(writer, sheet_name='Variable_List', index=False)
            
            # 回归结果
            if self.ordinal_results is not None:
                self.ordinal_results.to_excel(writer, sheet_name='Regression_Results', index=False)
        
        print(f"分析结果已保存至: {output_path}")
    
    def generate_report(self):
        """Print a structured plain-text summary of key analysis results."""
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS REPORT")
        print("="*60)
        
        print(f"\n【Sample Characteristics】")
        print(f"Total participants: {self.total_n}")
        print(f"Normal group: {self.normal_n} ({self.normal_n/self.total_n*100:.1f}%)")
        print(f"Risk group: {self.risk_n} ({self.risk_n/self.total_n*100:.1f}%)")
        
        if self.gender_var:
            print(f"\n【Gender Distribution】")
            gender_dist = self.df['p_gender_label'].value_counts()
            for gender, count in gender_dist.items():
                print(f"  {gender}: {count} ({count/self.total_n*100:.1f}%)")
        
        print(f"\n【Univariate Analysis】")
        print(f"Analyzed {len(self.categorical_vars)} categorical and {len(self.continuous_vars)} continuous variables")
        print(f"Found {len(self.significant_vars)} statistically significant variables (P<0.05)")
        
        if self.ordinal_results is not None:
            print(f"\n【Multivariate Ordinal Logistic Regression】")
            significant_or = self.ordinal_results[self.ordinal_results['P值'] < 0.05]
            print(f"Independent risk factors: {len(significant_or)}")
            
            for _, row in significant_or.iterrows():
                print(f"  • {row['变量']}: OR={row['OR值']} (95%CI: {row['95% CI 下限']}-{row['95% CI 上限']}), P={row['P值']}")
        
        print("\n" + "="*60)
        print("Tables saved in:", self.tables_dir)
        print("Figures saved in:", self.figures_dir)


def main():
    """Entry point: configure paths, run the full analysis pipeline, and save outputs."""
    # File paths
    input_file = r"D:\科研\ais_multimodal_multitask_pipeline\data\processed\merged_all_sheets_v2.xlsx"
    output_dir = r"D:\科研\ais_multimodal_multitask_pipeline\outputs\data_analyze\v4"
    
    try:
        print("="*60)
        print("ADOLESCENT IDIOPATHIC SCOLIOSIS RISK FACTOR ANALYSIS")
        print("="*60)
        
        # 初始化分析
        analysis = StatisticalAnalysis(input_file, output_dir)
        
        # 执行分析流程
        print("\n[Step 1/4] Descriptive statistics and baseline characteristics...")
        analysis.descriptive_statistics()
        
        print("\n[Step 2/4] Ordinal logistic regression analysis...")
        analysis.ordinal_logistic_regression()
        
        print("\n[Step 3/4] Creating publication-quality figures...")
        analysis.create_visualizations()
        
        print("\n[Step 4/4] Exporting results...")
        analysis.export_results()
        
        # 生成报告
        analysis.generate_report()
        
        print("\n✅ Analysis completed successfully!")
        print(f"📊 Results saved in: {output_dir}")
        
    except FileNotFoundError:
        print(f"❌ Error: File not found - {input_file}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()