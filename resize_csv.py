"""
Resize CSV file to exactly 16MB
"""

import pandas as pd
import os

def resize_csv_to_16mb(input_file, output_file, target_size_mb=16):
    """Resize CSV file to exactly 16MB"""
    
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Original file: {len(df):,} rows")
    original_size = os.path.getsize(input_file) / (1024 * 1024)
    print(f"Original size: {original_size:.2f} MB")
    
    # Calculate target rows (rough estimate)
    target_size_bytes = target_size_mb * 1024 * 1024
    bytes_per_row = original_size * 1024 * 1024 / len(df)
    target_rows = int(target_size_bytes / bytes_per_row)
    
    print(f"Target size: {target_size_mb} MB")
    print(f"Estimated target rows: {target_rows:,}")
    
    # Take only the required number of rows
    df_resized = df.head(target_rows)
    
    # Save resized file
    df_resized.to_csv(output_file, index=False)
    
    # Check actual size
    actual_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\nResized file: {len(df_resized):,} rows")
    print(f"Actual size: {actual_size:.2f} MB")
    
    # If still too large, reduce further
    if actual_size > target_size_mb:
        print("File still too large, reducing further...")
        reduction_factor = target_size_mb / actual_size
        new_rows = int(len(df_resized) * reduction_factor * 0.95)  # 5% buffer
        
        df_final = df_resized.head(new_rows)
        df_final.to_csv(output_file, index=False)
        
        final_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"Final file: {len(df_final):,} rows")
        print(f"Final size: {final_size:.2f} MB")
    
    return output_file

if __name__ == "__main__":
    input_file = "large_test_data_16mb.csv"
    output_file = "test_data_16mb.csv"
    
    if os.path.exists(input_file):
        resize_csv_to_16mb(input_file, output_file, 16)
        print(f"\n✅ Resized file saved as: {output_file}")
    else:
        print(f"❌ Input file not found: {input_file}")
