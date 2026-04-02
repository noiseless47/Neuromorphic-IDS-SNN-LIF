import os
import glob
import re
import numpy as np
import pandas as pd
from PyLTSpice import SimRunner, SpiceEditor, RawRead

def extract_spikes_from_raw(raw_file, threshold=0.5):
    """
    Parses an LTSpice .raw output file, reads the membrane voltage traces (V_mem),
    and counts how many times the comparator threshold fired (spike rate).
    """
    try:
        LTR = RawRead(raw_file)
        # Using typical node names like V(out1), V(out2), V(neuron_out)
        traces = LTR.get_trace_names()
        neuron_out_traces = [t for t in traces if 'out' in t.lower() and 'v(' in t.lower()]
        
        # If no specific output nodes found, just return empty to signal failure
        if not neuron_out_traces:
            return []
            
        extracted_features = []
        for trace_name in neuron_out_traces:
            voltage = LTR.get_trace(trace_name).get_wave()
            
            # Simple threshold crossing detection to count spikes
            spikes = 0
            is_high = False
            for v in voltage:
                if v > threshold and not is_high:
                    spikes += 1
                    is_high = True
                elif v < threshold * 0.8: # hysteresis
                    is_high = False
                    
            extracted_features.append(spikes)
            
        return extracted_features
    except Exception as e:
        print(f"Failed to parse trace {raw_file}: {e}")
        return []

def scrape_power_from_log(log_file):
    """
    Reads the LTSpice .log file to extract the average power dissipation.
    Requires a .meas directive in the SPICE netlist (e.g., .meas tran Avg_Power AVG V(Vcc)*I(Vcc))
    """
    try:
        if not os.path.exists(log_file):
            return 0.0
            
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            log_content = f.read()
            
        # Example matcher: avg_power = 0.00123 W
        # Your teammates must ensure a .meas directive prints the power to the log!
        match = re.search(r'(?i)avg_power\s*[=:]\s*([0-9\.eE+\-]+)', log_content)
        if match:
            return float(match.group(1))
        return None
    except Exception as e:
        print(f"Failed to parse log {log_file}: {e}")
        return None

def orchestrate_hardware_subset(circuit_file, subset_file, pwl_dir, results_dir="d:/Neuromorphic-IDS-SNN-LIF/spice_outputs"):
    """
    Automates the LTSpice execution for the HIL subset.
    """
    print("=== [STEP 5 & 6] HARDWARE ORCHESTRATION & EXTRACTION ===")
    
    if not os.path.exists(subset_file):
        print(f"Subset file {subset_file} not found. Did you run the pipeline?")
        return None
        
    df = pd.read_csv(subset_file)
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize the PyLTSpice Runner
    try:
        runner = SimRunner(output_folder=results_dir, simulator='LTspice')
        sim_ready = True
    except Exception as e:
        print(f"Warning: Could not initialize LTSpice SimRunner. Make sure LTSpice is installed. ({e})")
        sim_ready = False
    
    if not sim_ready or not os.path.exists(circuit_file):
        if not os.path.exists(circuit_file):
            print(f"Circuit file {circuit_file} not found! Please place your base .asc or .net file here.")
        
        print("\n[WARNING] Hardware execution unavailable. Generating dummy hardware results for software pipeline validation...")
        dummy_features = []
        for idx, row in df.iterrows():
            # Random spikes + attack category
            spikes = np.random.randint(0, 50, size=8).tolist() # 8 dummy output neurons
            dummy_features.append(spikes + [row['attack_cat']])
        
        out_df = pd.DataFrame(dummy_features)
        out_df.to_csv(os.path.join(results_dir, "hardware_features_output.csv"), index=False)
        return out_df.values
        
    netlist = SpiceEditor(circuit_file)
    
    print(f"Queuing {len(df)} SPICE simulations...")
    successful_runs = []
    
    # Batch run all the samples
    for idx, row in df.iterrows():
        cat_id = int(row['attack_cat'])
        run_name = f"sample_{idx}_cat_{cat_id}"
        
        # Here we tell the netlist to use the specific PWL files for this sample
        # Note: Your teammates' circuit must have voltage sources named V1, V2, etc., pointing to PWL files
        for i in range(1, 9): # Assuming 8 PCA features -> 8 PWL sources
            pwl_path = os.path.abspath(os.path.join(pwl_dir, f"category_{cat_id}", f"sample_{idx}_feature_{i}.txt"))
            # Example: V1 PWL(file="path")
            netlist.set_parameters(**{f"V_signal_{i}": f'PWL(file="{pwl_path}")'})
            
        # Run it (wait internally handles -b batch execution)
        runner.run_now(netlist, run_filename=run_name)
        successful_runs.append((run_name, cat_id))
    
    print("Simulations complete. Extracting spikes and power...")
    
    all_features = []
    total_power = 0.0
    power_samples = 0
    
    for run_name, cat_id in successful_runs:
        raw_file = os.path.join(results_dir, f"{run_name}.raw")
        log_file = os.path.join(results_dir, f"{run_name}.log")
        
        features = extract_spikes_from_raw(raw_file)
        power = scrape_power_from_log(log_file)
        
        if power is not None:
            total_power += power
            power_samples += 1
            
        all_features.append(features + [cat_id])
        
    avg_power = total_power / power_samples if power_samples > 0 else 0.0
    print(f"\n[METRIC] Average Hardware Inference Power: {avg_power * 1e3:.2f} mW")
    
    # Save the extracted features
    out_matrix = np.array(all_features)
    out_df = pd.DataFrame(out_matrix)
    out_path = os.path.join(results_dir, "hardware_features_output.csv")
    out_df.to_csv(out_path, index=False)
    
    print(f"Extracted hardware feature vectors saved to {out_path}")
    return out_matrix


if __name__ == "__main__":
    circuit = "d:/Neuromorphic-IDS-SNN-LIF/circuit/snn_2layer_base.asc"  # The file teammates will provide
    subset = "d:/Neuromorphic-IDS-SNN-LIF/processed_data/hil_subset_features.csv"
    pwl = "d:/Neuromorphic-IDS-SNN-LIF/encoded_spikes/population"
    
    orchestrate_hardware_subset(circuit, subset, pwl)
