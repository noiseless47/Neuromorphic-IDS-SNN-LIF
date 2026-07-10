import os
import pandas as pd
import numpy as np

def extract_spikes_from_csv(csv_path, target_columns=None, threshold=0.0005):
    """
    Reads a Cadence Virtuoso CSV file and counts spikes for given target columns.
    
    Args:
        csv_path (str): Path to the Cadence CSV.
        target_columns (list): List of column names to parse (e.g., ['/vout Y', '/vout2 Y']).
                               If None, it auto-detects columns with 'out' and 'Y'.
        threshold (float): Voltage threshold to count as a spike.
        
    Returns:
        dict: A dictionary mapping column names to their spike counts.
    """
    try:
        # Cadence CSVs usually have a single header row
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read {csv_path}: {e}")
        return {}

    # Auto-detect if targets not provided
    if target_columns is None:
        # Look for columns that represent Y-axis (voltage) and contain 'out' or 'vcom'
        target_columns = [col for col in df.columns if (' Y' in col) and ('out' in col.lower() or 'vcom' in col.lower() or 'net025' in col.lower())]
        
    if not target_columns:
        print(f"Warning: No target output columns found in {csv_path}.")
        print(f"Available columns: {df.columns.tolist()}")
        return {}

    results = {}
    for col in target_columns:
        if col not in df.columns:
            print(f"Column {col} not found in {csv_path}. Skipping.")
            continue
            
        voltages = df[col].values
        
        # Hysteresis spike detection
        spikes = 0
        is_high = False
        
        # We use a slightly lower threshold to reset the spike state
        reset_threshold = threshold * 0.8
        
        for v in voltages:
            if v > threshold and not is_high:
                spikes += 1
                is_high = True
            elif v < reset_threshold:
                is_high = False
                
        results[col] = spikes
        
    return results

def process_batch_cadence_csvs(csv_directory, output_array_path="features_array.npy", num_neurons=1):
    """
    Processes an entire folder of Cadence CSVs and outputs a features array for SVM.
    Assumes files are named in a way that indicates the sample index, or we just process them sorted.
    """
    csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {csv_directory}")
        return None
        
    features_list = []
    
    # Sort files to ensure deterministic ordering (e.g. sample_0, sample_1)
    # This might need a custom sort if filenames are 'sample_10' vs 'sample_2'
    csv_files.sort()
    
    for f in csv_files:
        full_path = os.path.join(csv_directory, f)
        spike_dict = extract_spikes_from_csv(full_path, threshold=0.5)
        
        # Sort columns to ensure consistent order (e.g. vout1, vout2, vout3)
        sorted_cols = sorted(spike_dict.keys())
        feature_row = [spike_dict[col] for col in sorted_cols]
        
        # Extract label from filename (e.g., sample_0_cat_1.csv -> label is 1)
        # 4_svm_classification.py expects the last column to be the label
        label = 0 # Default fallback
        if '_cat_' in f:
            try:
                # e.g. 'sample_0_cat_1.csv' -> '1.csv' -> '1'
                cat_str = f.split('_cat_')[-1].split('.')[0]
                label = int(cat_str)
            except ValueError:
                pass
                
        feature_row.append(label)
        features_list.append(feature_row)
        
    # Convert to numpy array
    features_array = np.array(features_list)
    
    if output_array_path:
        np.save(output_array_path, features_array)
        print(f"Saved parsed features for {len(features_list)} samples to {output_array_path}")
        
    return features_array

if __name__ == '__main__':
    # Run batch process on dummy outputs
    out_dir = 'dummy_cadence_outputs'
    if os.path.exists(out_dir):
        print(f"Processing all CSVs in {out_dir}...")
        process_batch_cadence_csvs(out_dir)
    else:
        # Fallback to test lif6.csv
        test_csv = 'lif6.csv'
        if os.path.exists(test_csv):
            print(f"Testing parser on {test_csv}...")
            results = extract_spikes_from_csv(test_csv, target_columns=['/vout Y', '/I8/net014 Y'])
            print("\nSpike counts extracted:")
            for col, count in results.items():
                print(f"  {col}: {count} spikes")
        else:
            print(f"Could not find {test_csv} to test.")
