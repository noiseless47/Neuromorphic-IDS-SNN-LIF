import os
import glob
import numpy as np
import pandas as pd

def parse_pwl_file(pwl_path):
    """Parses a PWL file into a list of (time, voltage) tuples."""
    data = []
    if not os.path.exists(pwl_path):
        return data
    with open(pwl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                data.append((float(parts[0]), float(parts[1])))
    return data

def interpolate_pwl(pwl_data, time_array):
    """Interpolates PWL data onto a uniform time array."""
    if not pwl_data:
        return np.zeros_like(time_array)
    pwl_t = np.array([p[0] for p in pwl_data])
    pwl_v = np.array([p[1] for p in pwl_data])
    # Piecewise linear interpolation
    return np.interp(time_array, pwl_t, pwl_v)

def simulate_lif_neuron(input_voltage_trace, dt, tau=0.005, threshold=0.8, rest_v=0.0, reset_v=0.0):
    """
    Simulates a very basic Leaky Integrate-and-Fire neuron.
    Since we are summing 4 PWL inputs (each 0 or 1.2V), the max input is 4.8V.
    """
    v_mem = rest_v
    v_out_trace = []
    
    # We want a 1.2V output spike that lasts a short duration when it fires
    spike_duration_steps = int(0.0001 / dt) # 0.1ms spike
    spike_countdown = 0
    
    for v_in in input_voltage_trace:
        if spike_countdown > 0:
            v_out_trace.append(1.2)
            spike_countdown -= 1
            v_mem = reset_v # Hold at reset while spiking
        else:
            v_out_trace.append(0.0)
            
            # Leaky integration
            # We scale the input up heavily so it easily crosses the threshold
            dv = (-(v_mem - rest_v) / tau + (v_in * 1000.0)) * dt
            v_mem += dv
            
            if v_mem >= threshold:
                spike_countdown = spike_duration_steps
                v_mem = reset_v
                
    return np.array(v_out_trace)

def main():
    pwl_dir = "pwl_sources"
    out_dir = "dummy_cadence_outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    # Find all sample directories that have the format sample_X_cat_Y
    sample_dirs = [d for d in os.listdir(pwl_dir) if 'sample_' in d and '_cat_' in d]
    
    if not sample_dirs:
        print(f"No proper sample directories found in {pwl_dir}.")
        return

    print(f"Found {len(sample_dirs)} samples to simulate.")
    
    # Define a simulation time array (e.g., 0 to 10ms with 10us steps)
    t_end = 0.01
    dt = 1e-5
    time_array = np.arange(0, t_end, dt)
    
    for s_dir in sample_dirs:
        full_s_dir = os.path.join(pwl_dir, s_dir)
        
        # We need to simulate 8 neurons
        output_data = {'Time X': time_array}
        
        for neuron_idx in range(8):
            # Sum the 4 inputs for this neuron
            total_input = np.zeros_like(time_array)
            for m in range(4):
                pwl_path = os.path.join(full_s_dir, f"V_neuron_{neuron_idx}_m_{m}.txt")
                pwl_data = parse_pwl_file(pwl_path)
                v_trace = interpolate_pwl(pwl_data, time_array)
                total_input += v_trace
                
            # Simulate the LIF hardware response
            v_out = simulate_lif_neuron(total_input, dt, tau=0.002, threshold=5.0)
            
            # Add to output
            output_data[f'/vout{neuron_idx} Y'] = v_out
            
        # Create DataFrame and save as Cadence-style CSV
        df = pd.DataFrame(output_data)
        out_csv_path = os.path.join(out_dir, f"{s_dir}.csv")
        df.to_csv(out_csv_path, index=False)
        print(f"Generated {out_csv_path}")
        
    print(f"Done. Successfully generated {len(sample_dirs)} simulated Cadence CSVs.")

if __name__ == "__main__":
    main()
