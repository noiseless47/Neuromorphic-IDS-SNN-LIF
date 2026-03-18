# Neuromorphic Intrusion Detection System (SNN + LIF + SVM) 🧠⚡

An interdisciplinary, novel framework for **Energy-Efficient Network Security** utilizing Analog Spiking Neural Networks (SNNs) driven by Leaky Integrate-and-Fire (LIF) circuits implemented in LTSpice/Cadence.

*Developed as an Interdisciplinary Project (IDP Phase 1) at RV College of Engineering.*

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LTSpice](https://img.shields.io/badge/SPICE-LTSpice%2FCadence-red.svg)
![Status](https://img.shields.io/badge/Status-Patent%20Pending-success.svg)

---

## 🎯 The Motivation
Traditional Machine Learning Intrusion Detection Systems (IDS) running 24/7 consume enormous power due to dense MAC operations (100–500 mW). By porting deep learning capabilities into the neuromorphic space via **Analog Spiking Circuits**, our hardware-in-the-loop methodology reduces this continuous monitoring power consumption to **under 1 mW**, solving the critical 24/7 security energy crisis.

---

## 🧬 The Revolutionary Methodology

This project bridges **Data Science** and **Analog VLSI Design** via an 8-step pipeline:

1. **Dataset Collection**: Loads the `UNSW-NB15` dataset consisting of real-world normal and attack networking signatures.
2. **Data Preprocessing & PCA**: The raw numerical landscape is min-max normalized, then highly compressed using **Principal Component Analysis (PCA)** to perfectly map massive data shapes onto restricted, physical Analog Neuron counts.
3. **Feature Extraction**: Distills the most crucial networking traits (packet rates, protocol byte sizes, etc.) from the PCA space.
4. **Spike Encoding**: Translates the normalized dataset coordinates into explicit binary timings using *Time-To-First-Spike (TTFS)*, *Rate Coding*, or *Gaussian Receptive Population Coding*.
   - Outputs: Direct **Piece-Wise Linear (PWL)** `.txt` files native to SPICE.
5. **Analog SNN Processing (Hardware)**: A physical 2-layer Analog SNN designed in **LTSpice / Cadence**. The PWL spike sets stream as input Voltages into an Op-Amp Integrator and Comparator framework representing the LIF logic.
6. **Spike-Based Feature Extraction**: Parses `.raw` outputs from LTSpice simulations back into python natively to extract real-world hardware firing statistics.
7. **Classification (SVM)**: Support Vector Machine trained specifically on the hardware spike responses to classify nodes accurately.
8. **Performance Evaluation**: Generates Accuracy, FPR, F1-Scores alongside direct SPICE-validated Nanowatt/Microwatt power figures.

---

## 📂 Repository Structure

```text
Neuromorphic-IDS-SNN-LIF/
│
├── raw datasets/               # Placeholder for UNSW-NB15 dataset chunks (Ignored in Git)
├── processed_data/             # Output directory for the PCA-scaled feature sets
├── pwl_sources/                # Auto-generated Piece-Wise Linear LTSpice inputs
├── spice_outputs/              # Destination for LTSpice .raw evaluation logs
│
├── 1_data_pipeline.py          # Data ingestion, Cleaning, MinMax scaling, and PCA Mapping
├── 2_spike_encoder.py          # Translates PCA targets to Rate/TTFS/Population SPICE PWLs
├── 3_analog_snn_runner.py      # Bridging pipeline reading hardware .raw signals -> arrays
├── 4_svm_classification.py     # SVM Model evaluating Physical Array data & displaying metrics
│
├── requirements.txt            # Python dependencies (pandas, sklearn, PyLTSpice)
└── README.md                   # Project Overview
```

## 🚀 Setting Up the Pipeline

### 1. Installation
Ensure you have Python 3.9+ and your target SPICE Simulator (LTSpice/Cadence) installed.
```bash
git clone <repository_url>
cd Neuromorphic-IDS-SNN-LIF
pip install -r requirements.txt
```

### 2. Loading the Data
Place the `UNSW-NB15` csvs into the `raw datasets/` folder.
Run the preprocessing pipeline to derive your dimension-reduced parameters:
```bash
python 1_data_pipeline.py
```

### 3. Generating Analog Spikes
Convert the abstract math directly into analog voltage pulse files:
```bash
python 2_spike_encoder.py
```
*This dumps standard `t v` syntax maps inside `pwl_sources/`.*

### 4. Running the Physical SNN
- Drag the respective `PWL` sources into your Cadence / LTSpice schematic inputs.
- Run a `.tran` transient simulation.
- Export or target the output node vectors to `.raw` logs.

### 5. Final Hardware Verification
Run the evaluation bridge to extract those real physical timings and classify:
```bash
python 3_analog_snn_runner.py  # If debugging the log ingestion explicitly
python 4_svm_classification.py # Trains and reports Accuracy / F1
```

---

## 🔒 Patents & Licensing
*All designs, analog implementations, and extraction techniques herein are developed per RV College of Engineering IDP protocols. Unauthorized distribution prior to patent filing is strictly prohibited.*
