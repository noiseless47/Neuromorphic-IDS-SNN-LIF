import os
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore') # Ignore sklearn warnings for cleaner output

# --- 1. SPICE & LIF SIMULATION MATH ---
TIME_STEPS = 10
TIMESTEP_DURATION = 0.001
VOLTAGE_HIGH = 1.2
VOLTAGE_LOW = 0.0

def rate_encoding(value, T=TIME_STEPS):
    spikes = np.random.rand(T) < value
    return spikes.astype(int)

def population_coding(value, M=4, T=TIME_STEPS):
    means = np.linspace(0, 1, M)
    sigma = 1.0 / (M - 1)
    spike_trains = []
    for mu in means:
        activation = np.exp(-0.5 * ((value - mu) / sigma)**2)
        spike_trains.append(rate_encoding(activation, T))
    return spike_trains

def translate_to_ltspice_pwl(spike_train, t_step=TIMESTEP_DURATION, v_high=VOLTAGE_HIGH, v_low=VOLTAGE_LOW):
    edges = []
    rise_fall = t_step * 0.01
    current_time = 0.0
    for val in spike_train:
        v_target = v_high if val == 1 else v_low
        edges.append((current_time, v_target))
        current_time += t_step - rise_fall
        edges.append((current_time, v_target))
    return edges

def interpolate_pwl(pwl_data, time_array):
    if not pwl_data: return np.zeros_like(time_array)
    pwl_t = np.array([p[0] for p in pwl_data])
    pwl_v = np.array([p[1] for p in pwl_data])
    return np.interp(time_array, pwl_t, pwl_v)

def simulate_lif_neuron(input_voltage_trace, dt, tau=0.002, threshold=5.0, rest_v=0.0, reset_v=0.0):
    v_mem = rest_v
    v_out_trace = []
    spike_duration_steps = int(0.0001 / dt)
    spike_countdown = 0
    for v_in in input_voltage_trace:
        if spike_countdown > 0:
            v_out_trace.append(1.2)
            spike_countdown -= 1
            v_mem = reset_v
        else:
            v_out_trace.append(0.0)
            dv = (-(v_mem - rest_v) / tau + (v_in * 1000.0)) * dt
            v_mem += dv
            if v_mem >= threshold:
                spike_countdown = spike_duration_steps
                v_mem = reset_v
    return np.array(v_out_trace)

def count_spikes_hysteresis(v_trace, high_thresh=0.8, low_thresh=0.2):
    spikes = 0
    is_high = False
    for v in v_trace:
        if v >= high_thresh and not is_high:
            spikes += 1
            is_high = True
        elif v <= low_thresh and is_high:
            is_high = False
    return spikes

# --- 2. LIVE INFERENCE ENGINE ---
def run_live_inference():
    print("\n" + "="*70)
    print("  NEUROMORPHIC IDS - LIVE HARDWARE INFERENCE ENGINE")
    print("="*70)
    
    models_dir = "models"
    try:
        le_cat = joblib.load(os.path.join(models_dir, "label_encoder_cat.joblib"))
        label_encoders = joblib.load(os.path.join(models_dir, "categorical_encoders.joblib"))
        scaler_raw = joblib.load(os.path.join(models_dir, "minmax_scaler_raw.joblib"))
        pca = joblib.load(os.path.join(models_dir, "pca_model.joblib"))
        scaler_analog = joblib.load(os.path.join(models_dir, "minmax_scaler_analog.joblib"))
        svm_model = joblib.load(os.path.join(models_dir, "svm_model.joblib"))
        raw_columns = joblib.load(os.path.join(models_dir, "raw_columns.joblib"))
    except Exception as e:
        print("Error loading models. Did you run step 1 and 4?")
        print(e)
        return

    # Grab a random raw packet from the original dataset
    dataset_path = "d:/Neuromorphic-IDS-SNN-LIF/raw datasets/UNSW-NB15_1.csv"
    print(f"\n[*] Intercepting a random live packet from {dataset_path}...")
    
    # We must read using the exact 49 columns that exist in the raw CSV
    COLUMNS = [
        'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 
        'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload', 'spkts', 'dpkts', 
        'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 
        'sjit', 'djit', 'stime', 'ltime', 'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat', 
        'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 
        'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 
        'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label'
    ]
    
    import random
    skip = sorted(random.sample(range(1, 500000), 490000)) 
    df_raw = pd.read_csv(dataset_path, header=None, names=COLUMNS, skiprows=skip, nrows=5000, low_memory=False)
    
    # Force it to find an attack packet for demonstration purposes
    df_attacks = df_raw[~df_raw['attack_cat'].isin(['Normal', ' ', '', np.nan])]
    
    if len(df_attacks) == 0:
        print("Couldn't find an attack in this random chunk. Run again!")
        return
        
    # Pick a random attack row
    random_idx = np.random.randint(0, len(df_attacks))
    packet = df_attacks.iloc[random_idx:random_idx+1].copy()
    
    true_label = packet['attack_cat'].values[0]
    if pd.isna(true_label) or true_label == ' ' or true_label == '':
        true_label = 'Normal'
    true_label = str(true_label).strip()
    
    # Feature Extraction mapping before heavy PCA operations
    dur_safe = packet['dur'].replace(0, 1e-6)
    packet['packet_rate'] = (packet['spkts'] + packet['dpkts']) / dur_safe
    
    # Remove binary 'label' and time identifiers.
    drop_cols = ['srcip', 'dstip', 'stime', 'ltime', 'label', 'attack_cat']
    for col in drop_cols:
        if col in packet.columns:
            packet = packet.drop(columns=[col])
            
    # Clean dataset
    packet.replace('-', np.nan, inplace=True)
    packet.fillna(0, inplace=True)
    
    print("[*] Raw Packet Intercepted & Cleaned.")
    
    # Preprocessing
    print("[*] Passing packet through PCA dimensionality reduction...")
    for col, le in label_encoders.items():
        # Handle unseen labels gracefully
        packet[col] = packet[col].astype(str).map(lambda s: s if s in le.classes_ else le.classes_[0])
        packet[col] = le.transform(packet[col])
        
    scaled_packet = scaler_raw.transform(packet)
    pca_packet = pca.transform(scaled_packet)
    analog_packet = scaler_analog.transform(pca_packet)
    
    pca_features = analog_packet[0]
    
    print("[*] Simulating Analog Neuromorphic Circuit (Population Coding & LIF Integration)...")
    
    t_end = 0.01
    dt = 1e-5
    time_array = np.arange(0, t_end, dt)
    spike_counts = []
    
    for i in range(8):
        val = pca_features[i]
        spike_trains = population_coding(val, M=4)
        
        total_input = np.zeros_like(time_array)
        for train in spike_trains:
            pwl_data = translate_to_ltspice_pwl(train)
            v_trace = interpolate_pwl(pwl_data, time_array)
            total_input += v_trace
            
        v_out = simulate_lif_neuron(total_input, dt)
        spikes = count_spikes_hysteresis(v_out)
        spike_counts.append(spikes)
        
    print(f"[*] Physical Hardware Output (Spike Counts): {spike_counts}")
    
    # SVM Prediction
    print("[*] Feeding hardware spikes into SVM Classifier...")
    features_array = np.array(spike_counts).reshape(1, -1)
    
    pred_idx = svm_model.predict(features_array)[0]
    
    # Map back to string
    class_mapping = {v: k for k, v in zip(le_cat.classes_, le_cat.transform(le_cat.classes_))}
    pred_label = class_mapping.get(pred_idx, "Unknown")
    
    print("\n" + "="*70)
    print("                      FINAL RESULT")
    print("="*70)
    print(f" TRUE PACKET TYPE : {true_label}")
    
    if pred_label == 'Normal':
        print(f" AI PREDICTION    : [ SAFE ] {pred_label} Traffic Detected")
    else:
        print(f" AI PREDICTION    : [ ALERT ] {pred_label} Attack Detected!")
    print("="*70)
    print("\n")

if __name__ == "__main__":
    run_live_inference()
