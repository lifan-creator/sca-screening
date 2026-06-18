"""
SCA Screening - Preprocessing: Questionnaire Data Cleaner
================================================================================
Removes invalid rows from the first-pass cleaned questionnaire file.
A row is considered invalid when every response field (all columns except
q_student_id) is either empty (NaN / blank string) or zero.

Input:  questionnaire_cleaned.xlsx
Output: questionnaire_cleaned_step2.xlsx

Author: Li Fan
Date: 2026-01-22
Version: 1.0
"""

import pandas as pd
import numpy as np

def clean_questionnaire_data(input_file, output_file=None):
    """
    Remove rows where all response fields (excluding q_student_id) are empty or zero.

    Parameters
    ----------
    input_file  : str  Path to the input Excel file.
    output_file : str  Path for the cleaned output file. Overwrites input if None.
    """
    # Load the questionnaire workbook
    print(f"正在读取文件: {input_file}")
    df = pd.read_excel(input_file)
    
    original_rows = len(df)
    print(f"原始数据行数: {original_rows}")
    
    # Identify response columns (all columns except the student ID)
    columns_to_check = [col for col in df.columns if col != 'q_student_id']

    # Build a boolean mask: True = row should be kept
    rows_to_keep = []

    for idx, row in df.iterrows():
        # Values in all response columns for this row
        values_to_check = row[columns_to_check]

        # Assume invalid until a meaningful (non-zero, non-empty) value is found
        all_empty_or_zero = True

        for value in values_to_check:
            # NaN and blank strings count as empty
            if pd.isna(value) or value == '':
                continue

            # Treat numeric zero (int, float, or string '0') as empty
            try:
                num_value = float(value)
                if num_value != 0:
                    all_empty_or_zero = False
                    break
            except (ValueError, TypeError):
                # Non-numeric, non-empty value — row contains real data
                if value is not None and value != '':
                    all_empty_or_zero = False
                    break

        # Keep rows with at least one meaningful response
        if not all_empty_or_zero:
            rows_to_keep.append(True)
        else:
            rows_to_keep.append(False)
            print(f"删除行 {idx+2} (Excel行号 {idx+2}): q_student_id = {row['q_student_id']}")

    # Apply the mask
    cleaned_df = df[rows_to_keep]
    
    cleaned_rows = len(cleaned_df)
    deleted_rows = original_rows - cleaned_rows
    
    print(f"\n处理完成!")
    print(f"保留行数: {cleaned_rows}")
    print(f"删除行数: {deleted_rows}")
    
    # Save the cleaned file
    if output_file is None:
        output_file = input_file
    
    cleaned_df.to_excel(output_file, index=False)
    print(f"结果已保存到: {output_file}")
    
    return cleaned_df

def main():
    """Entry point: run the questionnaire cleaner and print a brief summary."""
    input_file = "../new_dataset/questionnaire_cleaned.xlsx"
    output_file = "../new_dataset/questionnaire_cleaned_step2.xlsx"

    # Run cleaning
    cleaned_data = clean_questionnaire_data(input_file, output_file)

    # Print summary statistics
    if len(cleaned_data) > 0:
        print("\n前5行数据预览:")
        print(cleaned_data.head())

        # Student ID counts
        print(f"\nq_student_id 唯一值数量: {cleaned_data['q_student_id'].nunique()}")
        print(f"q_student_id 非空值数量: {cleaned_data['q_student_id'].count()}")

if __name__ == "__main__":
    main()