import os
import glob
import numpy as np
from PyLTSpice import RawRead

def extract_spikes_from_raw(raw_file, threshold=0.5):
    """
    Step 6: Spike-Based Feature Extraction
    Parses an LTSpice .raw output file, reads the membrane voltage traces (V_mem),
    and counts how many times the comparator threshold fired (spike rate).
    """
    try:
        LTR = RawRead(raw_file)
        time_steps = LTR.get_trace('time').get_time_axis()
        
        # In LTSpice, assuming you named the output neuron voltage nodes V(neuron_out_X)
        traces = LTR.get_trace_names()
        neuron_out_traces = [t for t in traces if 'neuron_out' in t.lower()]
        
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
        print(f"Failed to parse {raw_file}: {e}")
        return []

def batch_process_ltspice_outputs(results_dir="d:/Neuromorphic-IDS-SNN-LIF/spice_outputs"):
    """
    Iterates through all LTSpice simulation outputs and compiles the final Feature Vector matrix.
    """
    print("=== [STEP 6] SPIKE-BASED FEATURE EXTRACTION (SPICE LOGS) ===")
    raw_files = glob.glob(os.path.join(results_dir, "*.raw"))
    
    if not raw_files:
        print(f"No LTSpice .raw files found in {results_dir}.")
        print("Please run your Step 5 Analog SPICE simulation first!")
        return None
        
    all_features = []
    
    for raw in raw_files:
        print(f"Extracting spikes from {os.path.basename(raw)}...")
        features = extract_spikes_from_raw(raw)
        
        # Here we would normally extract the Label from the filename or a manifest map
        label = 1 if "attack" in raw.lower() else 0
        all_features.append(features + [label])
        
    print(f"Successfully extracted {len(all_features)} hardware-simulated feature vectors.")
    return np.array(all_features)

if __name__ == "__main__":
    batch_process_ltspice_outputs()
