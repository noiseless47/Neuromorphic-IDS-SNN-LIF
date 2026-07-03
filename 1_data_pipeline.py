import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
import joblib
import time

# Defining the standard 49 columns belonging to the UNSW-NB15 dataset files
COLUMNS = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 
    'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload', 'spkts', 'dpkts', 
    'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 
    'sjit', 'djit', 'stime', 'ltime', 'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat', 
    'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 
    'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 
    'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label'
]

def build_data_pipeline(data_path="d:/Neuromorphic-IDS-SNN-LIF/raw datasets", 
                        sample_size=None,
                        physical_neuron_count=8):
    """
    Step 1: Dataset Collection
    Step 2: Data Preprocessing (Cleaning + Min-Max + PCA)
    Step 3: Feature Extraction
    """
    print("=== [STEP 1] DATASET COLLECTION & EXTRACTION ===")
    start_time = time.time()
    
    files = glob.glob(os.path.join(data_path, "UNSW-NB15_*.csv"))
    if not files:
        raise FileNotFoundError("Could not locate UNSW-NB15 CSV files in the raw datasets directory.")
    
    dfs = []
    for f in files:
        print(f"Loading {os.path.basename(f)}...")
        df = pd.read_csv(f, names=COLUMNS, low_memory=False)
        dfs.append(df)
        
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded total dataset: {full_df.shape[0]} rows, {full_df.shape[1]} features.")
    
    # Feature Extraction mapping before heavy PCA operations
    print("\n=== [STEP 3] FEATURE EXTRACTION (Derived) ===")
    # Creating custom 'rate' column representing packets per second
    dur_safe = full_df['dur'].replace(0, 1e-6) # avoid /0
    full_df['packet_rate'] = (full_df['spkts'] + full_df['dpkts']) / dur_safe
    
    print("Selecting core categorical and numerical targets based on network behaviour...")
    # Remove binary 'label' and time identifiers. Keep 'attack_cat' for FULL Multi-class categorization.
    drop_cols = ['srcip', 'dstip', 'stime', 'ltime', 'label'] 
    clean_df = full_df.drop(columns=drop_cols)
    
    print("\n=== [STEP 2] DATA PREPROCESSING (Cleaning + Norm + PCA) ===")
    print("Cleaning dataset...")
    clean_df.replace('-', np.nan, inplace=True)
    
    # Ensure 'attack_cat' captures 'Normal' traffic properly instead of NaNs
    clean_df['attack_cat'] = clean_df['attack_cat'].replace(['', '-', ' ', np.nan, 0, '0'], 'Normal')
    clean_df['attack_cat'] = clean_df['attack_cat'].astype(str).str.strip().str.title()
    
    # Fill remaining numerical/categorical NaNs with 0
    clean_df.fillna(0, inplace=True) 
    
    # Downsample but force a PERFECTLY BALANCED dataset for the AI
    samples_per_class = 150
    print(f"Forcing perfectly balanced dataset with {samples_per_class} samples per attack category...")
    clean_df = clean_df.groupby('attack_cat', group_keys=False).apply(
        lambda x: x.sample(min(len(x), samples_per_class))
    ).sample(frac=1).reset_index(drop=True)
    
    le_cat = LabelEncoder()
    y = le_cat.fit_transform(clean_df['attack_cat'])
    class_mapping = dict(zip(le_cat.classes_, le_cat.transform(le_cat.classes_)))
    print(f"Multi-class mapped to integers: {class_mapping}")
    
    X_raw = clean_df.drop(columns=['attack_cat'])
    
    # Label Encode strings (protocol type, state, service)
    print("Encoding categorical protocol/states to numerical representation...")
    label_encoders = {}
    for col in X_raw.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        # Convert everything to string first in case of mixed types
        X_raw[col] = le.fit_transform(X_raw[col].astype(str))
        label_encoders[col] = le
    
    # Normalization (Min-Max)
    print("Applying Min-Max Normalization...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # PCA to reduce dimensions and match physical SPICE neuron count!
    print(f"Applying PCA dimensionality reduction to match target Analog Neurons (n={physical_neuron_count})...")
    pca = PCA(n_components=physical_neuron_count)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"PCA Variance Explained Ratio: {pca.explained_variance_ratio_}")
    print(f"Total Info Retained: {np.sum(pca.explained_variance_ratio_) * 100:.2f}%")
    
    # Reassemble and dump
    pca_columns = [f"pca_neuron_{i+1}" for i in range(physical_neuron_count)]
    final_df = pd.DataFrame(X_pca, columns=pca_columns)
    
    # We must apply another Min-Max scale because PCA can produce negative and unbounded values
    # Analog spikes mathematically operate from 0 to 1 ranges or precise voltage bounds (e.g. 0V to 1V)
    print("Re-normalizing PCA features strictly into [0, 1] voltage bounds for Spike Encoding...")
    analog_scaler = MinMaxScaler()
    final_df[pca_columns] = analog_scaler.fit_transform(final_df)
    
    final_df['attack_cat'] = y
    
    output_dir = "d:/Neuromorphic-IDS-SNN-LIF/processed_data"
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "pca_analog_features.csv")
    final_df.to_csv(out_file, index=False)
    
    # Save models for live inference
    models_dir = "d:/Neuromorphic-IDS-SNN-LIF/models"
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(le_cat, os.path.join(models_dir, "label_encoder_cat.joblib"))
    joblib.dump(label_encoders, os.path.join(models_dir, "categorical_encoders.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, "minmax_scaler_raw.joblib"))
    joblib.dump(pca, os.path.join(models_dir, "pca_model.joblib"))
    joblib.dump(analog_scaler, os.path.join(models_dir, "minmax_scaler_analog.joblib"))
    
    # Save the column names so inference knows what to expect
    joblib.dump(list(clean_df.drop(columns=['attack_cat']).columns), os.path.join(models_dir, "raw_columns.joblib"))
    
    # Generate HIL Subset (Hardware-In-the-Loop)
    hil_size = 500
    if len(final_df) > hil_size:
        print(f"\nGenerating Hardware-In-the-Loop (HIL) subset of size {hil_size} for LTSpice tests...")
        hil_df = final_df.groupby('attack_cat', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, int(hil_size * len(x) / len(final_df)))))
        ).sample(frac=1, random_state=42).reset_index(drop=True)
        hil_out_file = os.path.join(output_dir, "hil_subset_features.csv")
        hil_df.to_csv(hil_out_file, index=False)
        print(f"HIL Subset saved to: {hil_out_file}")
    
    print(f"\nPipeline completely mapped successfully in {time.time() - start_time:.2f} seconds!")
    print(f"Final Full Data saved to: {out_file}")
    print(final_df.head())

if __name__ == "__main__":
    # Feel free to adjust sample_size to None to process all 2.5M rows. 
    # Warning: PCA on 2.5M rows takes significant RAM. Defaulting to 100k subsets for testing.
    build_data_pipeline(sample_size=100000, physical_neuron_count=8)
