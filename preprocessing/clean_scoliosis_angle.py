"""
SCA Screening - Preprocessing: Scoliosis Angle Data Cleaner
================================================================================
Resolves duplicate student IDs in the scoliosis angle measurement sheet.

Deduplication rules:
  - Exact duplicate rows (all fields identical) → keep only the first occurrence
  - 2 rows with the same ID but different data → keep original ID; suffix second row with '01'
  - 3 rows with the same ID but different data → original ID, ID+'01', ID+'02'
  - More than 3 rows with the same ID → flag as exception and retain as-is

Input:  scoliosis_angle_cleaned.xlsx
Output: scoliosis_angle_cleaned_step2.xlsx

Author: Li Fan
Date: 2026-01-22
Version: 1.0
"""

import pandas as pd
import numpy as np

def handle_duplicate_student_ids(input_file, output_file):
    """
    Resolve duplicate student IDs in the scoliosis angle dataset.

    Parameters
    ----------
    input_file  : str  Path to the input Excel file.
    output_file : str  Path for the deduplicated output file.
    """
    
    print(f"正在读取文件: {input_file}")
    
    try:
        # Load the scoliosis angle workbook
        df = pd.read_excel(input_file)

        # Verify all required columns are present
        required_columns = [
            'sa_student_id',
            'sa_atr_max_angle', 
            'sa_max_atr_direction',
            'sa_max_atr_location',
            'sa_sa_percentage_of_maximum_angle_on_the_right_side',
            'sa_vertebral_position_of_maximum_angle',
            'sa_evaluation_result',
            'sa_calculation_name'
        ]
        
        # Abort if any required column is missing
        for col in required_columns:
            if col not in df.columns:
                print(f"错误: 列 '{col}' 不存在于数据中")
                print(f"数据中的列: {df.columns.tolist()}")
                return None
        
        # Log original data shape
        print(f"原始数据行数: {len(df)}")
        print(f"原始唯一学生ID数: {df['sa_student_id'].nunique()}")

        # Collect exception messages for IDs with >3 records
        exceptions = []

        # Accumulate processed rows for the output DataFrame
        processed_rows = []

        # Iterate over each unique student ID
        student_ids = df['sa_student_id'].unique()

        for student_id in student_ids:
            # Retrieve all records for this student
            student_records = df[df['sa_student_id'] == student_id].copy()

            # Single record — no deduplication needed
            if len(student_records) == 1:
                processed_rows.append(student_records.iloc[0].to_dict())
                continue

            record_count = len(student_records)

            # More than 3 records: flag as exception and retain as-is
            if record_count > 3:
                exceptions.append(f"学生ID {student_id} 有 {record_count} 条记录，超过3条限制")
                for i, record in student_records.iterrows():
                    processed_rows.append(record.to_dict())
                continue

            # Compare all fields except the student ID column
            comparison_cols = required_columns[1:]

            # Check whether all records are identical
            first_record = student_records.iloc[0]
            all_same = True
            for i in range(1, record_count):
                current_record = student_records.iloc[i]
                for col in comparison_cols:
                    # NaN-safe comparison
                    if pd.isna(first_record[col]) and pd.isna(current_record[col]):
                        continue
                    elif pd.isna(first_record[col]) or pd.isna(current_record[col]):
                        all_same = False
                        break
                    elif first_record[col] != current_record[col]:
                        all_same = False
                        break
                if not all_same:
                    break
            
            # Case 1: all records identical — keep only the first
            if all_same:
                print(f"学生ID {student_id}: {record_count} 条记录完全相同，只保留第一条")
                processed_rows.append(first_record.to_dict())

            # Case 2: records differ — retain first ID; suffix subsequent records
            else:
                print(f"学生ID {student_id}: {record_count} 条记录有差异，将修改重复记录的ID")

                # Keep the original ID for the first record
                first_record_dict = first_record.to_dict()
                processed_rows.append(first_record_dict)

                # Append numeric suffixes to subsequent records
                for i in range(1, record_count):
                    record_dict = student_records.iloc[i].to_dict()

                    # Suffix scheme based on total record count
                    if record_count == 2:
                        new_id = f"{student_id}01"
                    elif record_count == 3:
                        if i == 1:
                            new_id = f"{student_id}01"
                        else:  # i == 2
                            new_id = f"{student_id}02"

                    record_dict['sa_student_id'] = new_id
                    processed_rows.append(record_dict)

        # Build output DataFrame
        processed_df = pd.DataFrame(processed_rows)

        # Sort by student ID so suffixed rows appear adjacent to their originals
        processed_df = processed_df.sort_values('sa_student_id')

        # Save to Excel
        print(f"\n保存处理后的数据到: {output_file}")
        processed_df.to_excel(output_file, index=False)

        # Print summary statistics
        print("\n" + "="*50)
        print("数据处理完成！")
        print("="*50)
        print(f"处理后数据行数: {len(processed_df)}")
        print(f"处理后唯一学生ID数: {processed_df['sa_student_id'].nunique()}")
        
        # Report which student IDs were modified
        modified_ids = []
        for student_id in student_ids:
            if len(df[df['sa_student_id'] == student_id]) > 1:
                original_count = len(df[df['sa_student_id'] == student_id])
                processed_ids = processed_df[processed_df['sa_student_id'].astype(str).str.startswith(str(student_id))]['sa_student_id'].unique()
                if len(processed_ids) > 1:
                    modified_ids.append((student_id, original_count, processed_ids))
        
        if modified_ids:
            print("\n修改了ID的学生记录:")
            for student_id, original_count, new_ids in modified_ids:
                print(f"  学生ID {student_id}:")
                print(f"    原始记录数: {original_count}")
                print(f"    处理后ID: {list(new_ids)}")
        
        # Log and save any exceptions
        if exceptions:
            print("\n" + "="*50)
            print("异常情况:")
            print("="*50)
            for exception in exceptions:
                print(f"  {exception}")
            
            # Write exceptions to a text file for manual review
            exception_file = "data_processing_exceptions.txt"
            with open(exception_file, 'w', encoding='utf-8') as f:
                for exception in exceptions:
                    f.write(exception + "\n")
            print(f"\n异常信息已保存到: {exception_file}")
        
        # 显示处理后的数据示例
        print("\n" + "="*50)
        print("处理后数据示例:")
        print("="*50)
        print(processed_df.head(15).to_string())
        
        # 显示具体的重复记录处理示例
        print("\n" + "="*50)
        print("重复记录处理示例:")
        print("="*50)
        
        # 找出有重复的记录
        duplicate_ids = df['sa_student_id'].value_counts()
        duplicate_ids = duplicate_ids[duplicate_ids > 1].index.tolist()
        
        if duplicate_ids:
            for dup_id in duplicate_ids[:3]:  # 显示前3个示例
                print(f"\n学生ID {dup_id} 的处理:")
                
                # 原始数据
                original_data = df[df['sa_student_id'] == dup_id]
                print("原始数据:")
                print(original_data.to_string(index=False))
                
                # 处理后的数据
                processed_data = processed_df[processed_df['sa_student_id'].astype(str).str.startswith(str(dup_id))]
                print("\n处理后数据:")
                print(processed_data.to_string(index=False))
        
        return processed_df
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        return None
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Entry point: run the scoliosis angle deduplication pipeline."""
    # File paths
    input_file = "../new_dataset/scoliosis_angle_cleaned.xlsx"
    output_file = "../new_dataset/scoliosis_angle_cleaned_step2.xlsx"

    # Run deduplication
    result_df = handle_duplicate_student_ids(input_file, output_file)
    
    if result_df is not None:
        print("\n" + "="*50)
        print("处理规则总结:")
        print("="*50)
        print("1. 完全相同的重复记录 → 删除重复，只保留一条")
        print("2. 两条不同记录 → 第一条保持原ID，第二条ID加'01'")
        print("3. 三条不同记录 → 原ID, ID+01, ID+02")
        print("4. 超过三条记录 → 标记为异常，保持原样")
        print("="*50)

if __name__ == "__main__":
    main()