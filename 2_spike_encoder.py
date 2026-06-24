import numpy as np
import pandas as pd
import os

TIME_STEPS = 10
VOLTAGE_HIGH = 1.0 # 1V for a spike
VOLTAGE_LOW = 0.0  # 0V base
TIMESTEP_DURATION = 0.001 # 1 ms per simulation tick

def rate_encoding(value, T=TIME_STEPS):
    """ Rate coding: firing probability proportional to value """
    spikes = []
    freq = int(np.clip(value, 0, 1) * T)
    for t in range(T):
        spikes.append(1 if t < freq else 0)
    return spikes

def ttfs_encoding(value, T=TIME_STEPS):
    """ Time-To-First-Spike: Stronger value = earlier spike """
    spike_time = int((1.0 - np.clip(value, 0, 1)) * (T - 1))
    spikes = [1 if t == spike_time else 0 for t in range(T)]
    return spikes

def population_coding(value, M=4, T=TIME_STEPS):
    """
    Gaussian Receptive Fields Population Coding:
    A single input value (0 to 1) activates M distinct neurons to different degrees.
    Returns: a list of M spike trains (one for each neuron in the receptive field).
    """
    means = np.linspace(0, 1, M)
    sigma = 1.0 / (M - 1)
    
    spike_trains = []
    for mu in means:
        # Calculate activation using gaussian bell curve
        activation = np.exp(-0.5 * ((value - mu) / sigma)**2)
        
        # Apply rate encoding proportional to that gaussian activation
        spike_trains.append(rate_encoding(activation, T))
    return spike_trains

def translate_to_ltspice_pwl(spike_train, t_step=TIMESTEP_DURATION, v_high=VOLTAGE_HIGH, v_low=VOLTAGE_LOW):
    """
    Converts binary arrays [0, 1, 0] to LTSpice Piece-Wise Linear text definitions.
    E.g. 0.000 0.0\n0.001 1.0\n0.002 1.0\n0.003 0.0
    To create nice square pulses, we introduce a very small rise/fall time.
    """
    edges = []
    rise_fall = t_step * 0.01 # 1% of timestep for rising edge to avoid SPICE matrix singularity
    
    current_time = 0.0
    for val in spike_train:
        v_target = v_high if val == 1 else v_low
        
        # Start of timestep
        edges.append(f"{current_time:.6f} {v_target:.1f}")
        
        # End of timestep (flat line)
        current_time += t_step - rise_fall
        edges.append(f"{current_time:.6f} {v_target:.1f}")
        
        # Tiny Gap to next tick
        current_time += rise_fall
        
    return "\n".join(edges)

def generate_hardware_test_vectors(data_path="d:/Neuromorphic-IDS-SNN-LIF/processed_data/pca_analog_features.csv",
                                   output_dir="d:/Neuromorphic-IDS-SNN-LIF/pwl_sources",
                                   encoding_method="rate",
                                   num_samples_to_generate=5):
    """
    Reads the PCA data, encodes it, and dumps pure LTSpice PWL files!
    """
    print(f"=== [STEP 4] SPIKE ENCODING -> LTSpice PWL ({encoding_method.upper()}) ===")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Ensure Step 2 Data Pipeline runs first.")
        return
        
    df = pd.read_csv(data_path).head(num_samples_to_generate)
    
    os.makedirs(output_dir, exist_ok=True)
    
    pca_cols = [c for c in df.columns if 'pca' in c.lower()]
    
    # Normalize the PCA columns to [0, 1] range
    for col in pca_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if col_max != col_min:
            df[col] = (df[col] - col_min) / (col_max - col_min)
        else:
            df[col] = 0.5 # fallback flat value
    
    for i, row in df.iterrows():
        sample_attack_cat = int(row['attack_cat'])
        sample_dir = os.path.join(output_dir, f"sample_{i}_cat_{sample_attack_cat}")
        os.makedirs(sample_dir, exist_ok=True)
        
        neuron_idx = 0
        for col in pca_cols:
            val = row[col]
            
            if encoding_method == 'population':
                # This returns multiple trains
                trains = population_coding(val, M=4)
                for m_idx, tr in enumerate(trains):
                    pwl = translate_to_ltspice_pwl(tr)
                    with open(os.path.join(sample_dir, f"V_neuron_{neuron_idx}_m_{m_idx}.txt"), "w") as f:
                        f.write(pwl)
            else:
                if encoding_method == 'ttfs':
                    tr = ttfs_encoding(val)
                else:
                    tr = rate_encoding(val)
                
                pwl = translate_to_ltspice_pwl(tr)
                with open(os.path.join(sample_dir, f"V_neuron_{neuron_idx}.txt"), "w") as f:
                    f.write(pwl)
                    
            neuron_idx += 1
            
    print(f"Successfully generated PWL voltage sources for {num_samples_to_generate} samples in '{output_dir}'.")
    print("These `.txt` files can be directly attached to LTSpice voltage components `PWL(file=...)`.")

if __name__ == "__main__":
    generate_hardware_test_vectors(encoding_method="population", num_samples_to_generate=5)
