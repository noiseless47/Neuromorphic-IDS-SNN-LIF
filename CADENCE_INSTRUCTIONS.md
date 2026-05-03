Cadence / Virtuoso (Spectre / LTSpice) Handover Instructions

Purpose
-------
This document tells the Cadence engineer exactly what we have done so far, which files we are providing, how to connect the generated PWL inputs to the SNN schematic, exact simulation settings we recommend, and which output files we need back so our Python pipeline can ingest them automatically.

What we have done (on our side)
-------------------------------
- Preprocessed the UNSW-NB15 dataset and produced PCA-mapped analog features (8 PCA components).
  - File: processed_data/pca_analog_features.csv
- Created a small Hardware-in-the-Loop (HIL) subset used for running on SPICE.
  - File: processed_data/hil_subset_features.csv
- Generated LTSpice-compatible PWL files for N samples under pwl_sources/.
  - Format: pwl_sources/sample_<sample_idx>_cat_<attack_cat>/V_neuron_*.txt
  - Each file contains whitespace-separated time (s) and voltage (V) pairs.
- Updated pipeline to expect hardware outputs written to spice_outputs/hardware_features_output.csv.

Files we will hand you (already in repo)
-----------------------------------------
- All PWL files (folder): pwl_sources/
- HIL subset list (mapping + labels): processed_data/hil_subset_features.csv
- PCA features (for reference): processed_data/pca_analog_features.csv
- Example base circuit (if you prefer to place PWLs into this): circuit/snn_2layer_base.asc (may be missing — if so please tell us where to place PWLs)
- Please return: spice_outputs/*.raw (or CSV/ASCII export) and spice_outputs/*.log for each run, and the resulting spice_outputs/hardware_features_output.csv (or the raw data we can parse into it).

Which samples to run
--------------------
- Preferably run the full HIL subset listed in processed_data/hil_subset_features.csv.
- If time-limited, run the first 5 samples (pwl_sources/sample_0_cat_*/ ... ) and return results for those runs.

How PWL files map to the circuit (example)
------------------------------------------
Our encoder created one PWL per input neuron (for population encoding there are additional files named V_neuron_<n>_m_<m>.txt).

You should attach PWLs to the circuit input voltage sources. Two common approaches:
- LTSpice: Use a voltage source instance with `PWL FILE="<path>"` property.
  Example netlist line (LTSpice):

  V1 in1 0 PWL FILE="d:/Neuromorphic-IDS-SNN-LIF/pwl_sources/sample_0_cat_6/V_neuron_0.txt"

- Spectre (Cadence) / Virtuoso: Use a `vpwl` or `vsource` instance and set its `file` or `value` property.
  Example Spectre-style snippet (approximate):

  V1 (in1 0) vsource type=pwl file="d:/Neuromorphic-IDS-SNN-LIF/pwl_sources/sample_0_cat_6/V_neuron_0.txt"

If your schematic uses parameterized source names (recommended), please bind them so we can automate runs. Example parameter mapping used in our orchestrator code:
- Parameter names expected: `V_signal_1`, `V_signal_2`, ..., `V_signal_8` (one per PCA feature). If it's easier, replace or alias your voltage source instance properties so that the netlist contains something equivalent to `V1 PWL(file="...")`, `V2 PWL(file="...")`, etc.

Suggested node naming convention
--------------------------------
- Input nodes: `in1, in2, ..., in8` (or whatever names you have) — please include the exact netlist node names in your response.
- Output nodes to capture: any node(s) representing comparator outputs or neuron outputs (e.g., `out1`, `out2`, `neuron_out1`...). Provide the exact node names used in the netlist so our parser can find them.

Simulation settings (recommended)
---------------------------------
- Simulator: Spectre (Virtuoso) or LTSpice (we can accept either), but please export ASCII time-voltage data in a simple table if possible.
- Analysis: Transient.
- Stop time: equal to the final timestamp in the PWL files (default encoder uses 10 timesteps * 1 ms = 0.01 s). Please confirm the last timestamp in the chosen PWLs (example default: `0.01` seconds).
- Maximum timestep (max step): use a small fraction of the pulse width (recommended `1e-6` s to `1e-5` s). This keeps pulse edges resolved.
- Save/Print: configure outputs to write node voltages for all selected output nodes for the entire simulation time.

Power measurement (important)
-----------------------------
- Our `3_hardware_orchestrator.py` will try to parse the `.log` file for a line like:

  avg_power = 0.00123

  so please include a measurement directive in your netlist/run that prints an `avg_power` value into the run log. Example directives:

- LTSpice example (put in the netlist):

  .meas TRAN Avg_Power PARAM avg(V(Vcc)*I(Vcc))
  .measure print

  To ensure the log contains `avg_power = <value>`, you can add a line in the netlist .step or use .echo in a control statement to write a formatted string — if that is awkward, include the measured Avg_Power value in the .log with a clearly labeled line such as `avg_power = 0.000123`.

- Spectre (Cadence) example:

  Use `measure` statements or `-print` commands to write the average power to the run log. If you prefer, compute average power externally from node voltage and current exports, but please then write the computed value into the log with the text `avg_power = <value>` so our parser finds it.

Output format we need back
-------------------------
For each sample run `sample_<idx>_cat_<cat>` we need:
- `sample_<idx>_cat_<cat>.raw` (LTSpice raw OK) OR
a CSV/ASCII file with the header and columns where the first column is `time` and subsequent columns are node voltages. Example (CSV):

  time,out1,out2,out3,out4
  0.000000,0.0,0.0,0.0,0.0
  0.001000,1.0,0.0,0.0,1.0

- `sample_<idx>_cat_<cat>.log` containing the `avg_power = ...` line.
- The netlist or schematic version used for that run (so we can reproduce and map node names).

How we'll consume your outputs
-----------------------------
- Our `3_hardware_orchestrator.py` expects to find `.raw` and `.log` files in `spice_outputs/` named exactly `sample_<idx>_cat_<cat>.raw` and `.log`. If you cannot produce `.raw`, provide CSV/ASCII per-run files and we will adapt the reader.
- After processing, `3_hardware_orchestrator.py` will generate `spice_outputs/hardware_features_output.csv` with the extracted spike counts and labels.

Example “one-sample” run flow (what we expect you to do)
-----------------------------------------------------
1. Open `circuit/snn_2layer_base.asc` in Virtuoso or LTSpice.
2. Replace/point the 8 input voltage sources to:
   - `d:/Neuromorphic-IDS-SNN-LIF/pwl_sources/sample_0_cat_6/V_neuron_0.txt`
   - `d:/Neuromorphic-IDS-SNN-LIF/pwl_sources/sample_0_cat_6/V_neuron_1.txt`
   - ... up to the number of inputs your schematic expects
3. Set transient analysis stop time to 0.01 s (or the PWL final time) and maxstep to 1e-6.
4. Add a `.meas` / Spectre `measure` to compute average power and make sure it writes a line `avg_power = <value>` to the run log.
5. Run the simulation and export time/voltage traces for the output nodes as ASCII/CSV or save the `.raw` file from LTSpice.
6. Name the output files using the run name pattern `sample_0_cat_6.raw` and `sample_0_cat_6.log` (or `.csv` if exporting ASCII table).
7. Upload the files into `spice_outputs/` in the repository (or send them back to us), and provide the netlist/schematic used and the node mapping used for outputs.

Netlist snippets (copy/paste)
-----------------------------
- LTSpice input source example:

  V1 in1 0 PWL FILE="d:/Neuromorphic-IDS-SNN-LIF/pwl_sources/sample_0_cat_6/V_neuron_0.txt"
  V2 in2 0 PWL FILE="d:/Neuromorphic-IDS-SNN-LIF/pwl_sources/sample_0_cat_6/V_neuron_1.txt"

- Spectre input source example (approximate):

  Xv1 in1 0 vpwl file="d:/Neuromorphic-IDS-SNN-LIF/pwl_sources/sample_0_cat_6/V_neuron_0.txt"

(If your Spectre library uses a different symbol name for PWL sources we can adapt to the actual symbol you use.)

What we need you to return (summary)
------------------------------------
- For each sample run: `<runname>.raw` or `<runname>.csv` (ASCII table), and `<runname>.log` containing `avg_power = ...`.
- The netlist/schematic used for the run and the mapping of PWL filenames → voltage source instance names → schematic node names.
- If possible, a single consolidated CSV for quick inspection: `spice_outputs/hardware_features_output.csv` containing extracted spike counts and the sample label column (we can also create this file if you prefer to return raw CSVs and logs).

Contact / Notes
----------------
- We ran the pipeline up to the PWL generation for you. If you need different samples or different encoding type (rate/ttfs/population), tell us and we will regenerate PWLs.
- If you prefer, we can modify `3_hardware_orchestrator.py` to exactly match whatever naming scheme you have for runs and outputs — just return one example run and we will adapt.

Thank you — please drop the run artifacts into `spice_outputs/` or attach them to your message and we will proceed to ingest and run classification.
