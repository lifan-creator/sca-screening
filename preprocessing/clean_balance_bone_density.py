"""
SCA Screening - Preprocessing: Balance & Bone Density Sheet Cleaner
================================================================================
Reads DataSet202601022_cleaned_manually.xlsx and applies automated cleaning
across 11 sheets: 1 bone_density sheet and 10 balance condition sheets.

Per-sheet cleaning steps:
  1. Strip whitespace; replace empty strings with NaN
  2. Drop rows with a missing student ID or all-empty data fields
  3. Remove exact duplicate rows; append numeric suffixes to non-identical
     rows that share the same student ID
  4. Coerce data columns to numeric where feasible
Outputs the cleaned workbook as 11_sheet_cleaned.xlsx.

Author: Li Fan
Date: 2026-01-22
Version: 1.0
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def get_sheet_config(sheet_name):
    """
    Return sheet-specific config: the student ID column name and column prefix.
    """
    configs = {
        'bone_density': {
            'id_column': 'bd_student_id',
            'prefix': 'bd'
        },
        'balance_standing_natural': {
            'id_column': 'bsn_student_id',
            'prefix': 'bsn'
        },
        'balance_standing_upright': {
            'id_column': 'bsu_student_id',
            'prefix': 'bsu'
        },
        'balance_standing_phone': {
            'id_column': 'bsp_student_id',
            'prefix': 'bsp'
        },
        'balance_standing_eyes_closed': {
            'id_column': 'bsec_student_id',
            'prefix': 'bsec'
        },
        'balance_sitting_natural': {
            'id_column': 'bsn_student_id',
            'prefix': 'bsitn'
        },
        'balance_sitting_upright': {
            'id_column': 'bsu_student_id',
            'prefix': 'bsitu'
        },
        'balance_sitting_cross_leg': {
            'id_column': 'bscl_student_id',
            'prefix': 'bscl'
        },
        'balance_sitting_backrest': {
            'id_column': 'bsb_student_id',
            'prefix': 'bsitb'
        },
        'balance_sit_to_stand': {
            'id_column': 'bsts_student_id',
            'prefix': 'bsts'
        },
        'balance_foam': {
            'id_column': 'bf_student_id',
            'prefix': 'bf'
        }
    }
    
    return configs.get(sheet_name, {
        'id_column': 'student_id',
        'prefix': 'student'
    })

def clean_sheet_data(df, sheet_name, config):
    """
    Clean a single sheet: remove empty/duplicate rows and normalise numeric columns.
    Returns the cleaned DataFrame and a statistics dict.
    """
    id_column = config['id_column']
    prefix = config['prefix']

    # Initialise per-sheet cleaning statistics
    sheet_stats = {
        'sheet_name': sheet_name,
        'original_rows': len(df),
        'empty_rows_removed': 0,
        'duplicate_rows_removed': 0,
        'id_modified_count': 0,
        'final_rows': 0
    }
    
    if len(df) == 0:
        print(f"  ⚠️ Sheet '{sheet_name}' 为空，跳过处理")
        sheet_stats['final_rows'] = 0
        return df, sheet_stats
    
    print(f"  📋 处理 {sheet_name}: 原始 {len(df)} 行，ID列: {id_column}")
    
    # Preserve original row order for stable deduplication
    df = df.copy()
    df['_original_index'] = range(len(df))
    df['_original_id'] = df[id_column].copy()

    # 1. Normalise strings: strip whitespace and replace empty strings with NaN
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # 2. Drop rows with missing student ID or all-empty data fields
    rows_to_drop = []

    for idx, row in df.iterrows():
        # Skip rows with missing student ID
        student_id = row.get(id_column)
        if pd.isna(student_id) or (isinstance(student_id, str) and student_id.strip() == ''):
            rows_to_drop.append(idx)
            continue

        # Skip rows where every data column (excluding ID and helper columns) is empty
        other_columns = [col for col in df.columns if col not in [id_column, '_original_index', '_original_id']]
        if len(other_columns) > 0:
            other_values = row[other_columns]
            if other_values.isna().all() or (other_values.astype(str).str.strip() == '').all():
                rows_to_drop.append(idx)
    
    if rows_to_drop:
        df_before = len(df)
        df = df.drop(rows_to_drop).reset_index(drop=True)
        sheet_stats['empty_rows_removed'] = df_before - len(df)
    
    if len(df) == 0:
        print(f"  ⚠️ Sheet '{sheet_name}' 所有行均为空，全部删除")
        sheet_stats['final_rows'] = 0
        return pd.DataFrame(), sheet_stats
    
    # 3. Restore original row order
    df = df.sort_values('_original_index').reset_index(drop=True)

    # 4. Detect and resolve duplicate student IDs
    duplicate_counts = df[id_column].value_counts()
    duplicate_ids = duplicate_counts[duplicate_counts > 1].index.tolist()

    if duplicate_ids:
        print(f"    发现 {len(duplicate_ids)} 个重复ID")

        # Collect row indices to remove
        rows_to_remove = []
        # Map row indices to new (suffixed) IDs
        id_mapping = {}

        for student_id in duplicate_ids:
            # Retrieve all rows sharing this student ID
            mask = df[id_column] == student_id
            duplicate_rows = df[mask]

            if len(duplicate_rows) <= 1:
                continue

            # Compare all fields except helper columns
            compare_cols = [col for col in duplicate_rows.columns
                          if col not in ['_original_index', '_original_id']]

            # Group rows to identify exact duplicates
            grouped = duplicate_rows.groupby(compare_cols)

            for group_key, group_df in grouped:
                if len(group_df) > 1:
                    # Case A: exact duplicate — keep the first occurrence
                    rows_to_keep = group_df.index[0]
                    rows_to_drop_group = group_df.index[1:].tolist()
                    rows_to_remove.extend(rows_to_drop_group)
                    sheet_stats['duplicate_rows_removed'] += len(rows_to_drop_group)

            # Handle remaining rows with the same ID but differing data
            remaining_rows = df[mask & ~df.index.isin(rows_to_remove)]

            if len(remaining_rows) > 1:
                # Case B: same ID, different data — append a numeric suffix to disambiguate
                remaining_rows = remaining_rows.sort_values('_original_index')

                for i, row_idx in enumerate(remaining_rows.index, 1):
                    new_id = f"{student_id}{i:02d}"
                    id_mapping[row_idx] = new_id
                    sheet_stats['id_modified_count'] += 1

        # Apply the suffix-based ID remapping
        for idx, new_id in id_mapping.items():
            if idx in df.index:
                df.at[idx, id_column] = new_id

        # Drop exact-duplicate rows
        if rows_to_remove:
            df = df.drop(rows_to_remove).reset_index(drop=True)

    # 5. Coerce data columns to numeric where possible
    for col in df.columns:
        if col not in [id_column, '_original_index', '_original_id']:
            try:
                original_non_null = df[col].notna().sum()
                df[col] = pd.to_numeric(df[col], errors='coerce')

                # Revert if conversion causes excessive data loss (>50% of non-null values)
                if df[col].notna().sum() < original_non_null * 0.5:
                    df[col] = df[col].astype(str)
            except:
                # Conversion failed; retain original type
                pass

    # 6. Remove helper columns added during processing
    if '_original_index' in df.columns:
        df = df.drop('_original_index', axis=1)
    if '_original_id' in df.columns:
        df = df.drop('_original_id', axis=1)

    # 7. Record final row count
    sheet_stats['final_rows'] = len(df)
    
    return df, sheet_stats

def clean_all_sheets():
    """
    Orchestrate cleaning for all 11 sheets and write the output workbook.
    Returns True on success, False on failure.
    """
    # File paths
    input_file = "../new_dataset/DataSet202601022_cleaned_manually.xlsx"
    output_file = "../new_dataset/11_sheet_cleaned.xlsx"

    # Sheets to process
    sheet_names = [
        'bone_density',
        'balance_standing_natural',
        'balance_standing_upright',
        'balance_standing_phone',
        'balance_standing_eyes_closed',
        'balance_sitting_natural',
        'balance_sitting_upright',
        'balance_sitting_cross_leg',
        'balance_sitting_backrest',
        'balance_sit_to_stand',
        'balance_foam'
    ]
    
    # Initialise aggregate statistics
    overall_stats = {
        'total_sheets': len(sheet_names),
        'processed_sheets': 0,
        'failed_sheets': 0,
        'total_original_rows': 0,
        'total_empty_removed': 0,
        'total_duplicate_removed': 0,
        'total_id_modified': 0,
        'total_final_rows': 0
    }
    
    # Collect cleaned DataFrames and per-sheet statistics
    cleaned_data = {}
    sheet_statistics = []

    try:
        print("=" * 60)
        print("开始执行多sheet数据清洗脚本")
        print("=" * 60)

        # Record start time
        start_time = datetime.now()

        # 1. Validate input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
        
        print(f"📂 输入文件: {input_file}")
        print(f"📂 输出文件: {output_file}")
        print(f"📊 要处理的sheet数量: {len(sheet_names)}")
        print("-" * 60)
        
        # 2. Discover available sheets in the workbook
        excel_file = pd.ExcelFile(input_file)
        available_sheets = excel_file.sheet_names

        print(f"📄 文件中实际存在的sheet: {len(available_sheets)} 个")

        # 3. Process each sheet
        for sheet_name in sheet_names:
            print(f"\n🔍 处理Sheet: {sheet_name}")

            if sheet_name not in available_sheets:
                print(f"  ❌ Sheet '{sheet_name}' 不存在于文件中，跳过")
                overall_stats['failed_sheets'] += 1
                continue

            try:
                # Load sheet (all as strings to avoid automatic type coercion)
                df = pd.read_excel(input_file, sheet_name=sheet_name, dtype=str)

                # Retrieve sheet-specific configuration
                config = get_sheet_config(sheet_name)

                # Run sheet cleaner
                cleaned_df, sheet_stats = clean_sheet_data(df, sheet_name, config)

                # Store results
                cleaned_data[sheet_name] = cleaned_df
                sheet_statistics.append(sheet_stats)

                # Accumulate aggregate statistics
                overall_stats['processed_sheets'] += 1
                overall_stats['total_original_rows'] += sheet_stats['original_rows']
                overall_stats['total_empty_removed'] += sheet_stats['empty_rows_removed']
                overall_stats['total_duplicate_removed'] += sheet_stats['duplicate_rows_removed']
                overall_stats['total_id_modified'] += sheet_stats['id_modified_count']
                overall_stats['total_final_rows'] += sheet_stats['final_rows']
                
                print(f"  ✅ 完成: 原始 {sheet_stats['original_rows']} → 最终 {sheet_stats['final_rows']} 行")
                
            except Exception as e:
                print(f"  ❌ 处理Sheet '{sheet_name}' 时出错: {str(e)}")
                overall_stats['failed_sheets'] += 1
                import traceback
                traceback.print_exc()
        
        # 4. Write cleaned sheets to output workbook
        if cleaned_data:
            print(f"\n{'='*60}")
            print("💾 正在保存清洗后的数据到新文件...")

            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                for sheet_name, df in cleaned_data.items():
                    if len(df) > 0:
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        print(f"  ✅ 保存Sheet: {sheet_name} ({len(df)} 行)")
                    else:
                        # Preserve sheet structure even when all rows were removed
                        empty_df = pd.DataFrame(columns=df.columns)
                        empty_df.to_excel(writer, sheet_name=sheet_name, index=False)
                        print(f"  ⚠️ 保存空Sheet: {sheet_name}")
            
            print(f"\n✅ 数据已成功保存到: {output_file}")
        
        # 5. 打印详细统计信息
        print(f"\n{'='*60}")
        print("📊 数据清洗完成！详细统计信息：")
        print('='*60)
        
        # 总体统计
        print(f"\n📈 总体统计:")
        print(f"   成功处理的sheet: {overall_stats['processed_sheets']}/{overall_stats['total_sheets']}")
        print(f"   失败的sheet: {overall_stats['failed_sheets']}")
        print(f"   总原始记录数: {overall_stats['total_original_rows']}")
        print(f"   总删除空行数: {overall_stats['total_empty_removed']}")
        print(f"   总删除重复行数: {overall_stats['total_duplicate_removed']}")
        print(f"   总修改ID数: {overall_stats['total_id_modified']}")
        print(f"   总最终记录数: {overall_stats['total_final_rows']}")
        
        # 各sheet详细统计
        print(f"\n📋 各Sheet详细统计:")
        print("-" * 80)
        print(f"{'Sheet名称':<30} {'原始行数':<10} {'删除空行':<10} {'删除重复':<10} {'修改ID':<10} {'最终行数':<10}")
        print("-" * 80)
        
        for stats in sheet_statistics:
            print(f"{stats['sheet_name']:<30} "
                  f"{stats['original_rows']:<10} "
                  f"{stats['empty_rows_removed']:<10} "
                  f"{stats['duplicate_rows_removed']:<10} "
                  f"{stats['id_modified_count']:<10} "
                  f"{stats['final_rows']:<10}")
        
        # Report elapsed time and data retention rate
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\n⏱️  总耗时: {duration}")
        
        if overall_stats['total_original_rows'] > 0:
            efficiency = (overall_stats['total_final_rows'] / overall_stats['total_original_rows']) * 100
            print(f"📈 数据保留率: {efficiency:.2f}%")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("请确保 'DataSet202601022_cleaned_manually.xlsx' 文件存在于当前目录")
        return False
    
    except Exception as e:
        print(f"\n❌ 清洗过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Entry point: verify dependencies and run the multi-sheet cleaning pipeline.
    """
    print("=" * 60)
    print("         多Sheet Excel数据清洗工具")
    print("=" * 60)
    print("功能说明:")
    print("1. 处理11个sheet的数据清洗")
    print("2. 每个sheet有独立的学生ID列名")
    print("3. 处理重复数据和空行")
    print("4. 统一数据类型")
    print("5. 生成新的Excel文件")
    print("=" * 60)
    
    # Verify required packages are installed
    try:
        import pandas as pd
        import openpyxl
        import numpy as np
        print("✅ 所有依赖库已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖库: {e}")
        print("请使用以下命令安装:")
        print("pip install pandas openpyxl numpy")
        return
    
    # Run the cleaning pipeline
    success = clean_all_sheets()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 脚本执行成功！")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ 脚本执行失败，请检查错误信息")
        print("=" * 60)

if __name__ == "__main__":
    main()