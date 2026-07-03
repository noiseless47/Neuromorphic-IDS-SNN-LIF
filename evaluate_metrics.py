import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import os

# 1. Evaluate Neuromorphic Power
hw_features_path = "features_array.npy"
analog_energy_mJ = 0
if os.path.exists(hw_features_path):
    features_array = np.load(hw_features_path)
    # The first 8 columns are the spike counts for the 8 physical neurons
    total_spikes = np.sum(features_array[:, :-1])
    num_samples = len(features_array)
    avg_spikes_per_packet = total_spikes / num_samples
    
    # Established 65nm Analog LIF energy is ~45 pJ per spike
    energy_per_spike_pJ = 45.0
    analog_energy_per_packet_pJ = avg_spikes_per_packet * energy_per_spike_pJ
    analog_energy_mJ = analog_energy_per_packet_pJ / 1e9 # Convert pJ to mJ
    
    print("\n=== NEUROMORPHIC HARDWARE POWER PROFILE ===")
    print(f"Average Spikes per Packet : {avg_spikes_per_packet:.2f} spikes")
    print(f"Energy per Packet         : {analog_energy_per_packet_pJ:.2f} pJ ({analog_energy_mJ:.8f} mJ)")

# 2. Digital CPU Random Forest Comparison
data_path = "processed_data/pca_analog_features.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    # Sample down to the same 2242 sizes we used to keep it fair
    df = df.sample(min(2242, len(df)), random_state=42)
    X = df.drop(columns=['attack_cat']).values
    y = df['attack_cat'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Measure inference time
    start_time = time.perf_counter()
    y_pred = rf.predict(X_test)
    end_time = time.perf_counter()
    
    inference_time_sec = (end_time - start_time) / len(X_test)
    
    digital_acc = accuracy_score(y_test, y_pred)
    
    # Standard embedded CPU TDP ~ 15 Watts (15 Joules / second)
    cpu_tdp_watts = 15.0
    digital_energy_per_packet_joules = cpu_tdp_watts * inference_time_sec
    digital_energy_mJ = digital_energy_per_packet_joules * 1000.0
    
    print("\n=== CONVENTIONAL DIGITAL CPU CLASSIFIER (Random Forest) ===")
    print(f"Digital Multi-Class Accuracy : {digital_acc * 100:.2f} %")
    print(f"Average Inference Time       : {inference_time_sec * 1000.0:.4f} ms")
    print(f"Energy per Packet            : {digital_energy_mJ:.4f} mJ")
    
    if analog_energy_mJ > 0:
        efficiency_gain = digital_energy_mJ / analog_energy_mJ
        print(f"\n=> The Neuromorphic Pipeline is {efficiency_gain:,.0f}x more energy efficient than the digital CPU!")
