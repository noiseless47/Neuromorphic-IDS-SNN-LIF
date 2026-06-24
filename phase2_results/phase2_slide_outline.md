# Phase 2 Presentation Outline

## Slide 1: Title

Neuromorphic Intrusion Detection System using SNN and LIF Neurons

Team:
Asish Kumar Yeleti, Disha A, Joel Saha, Apoorva Yeshwanthraya Bilagundi

## Slide 2: Problem

Conventional IDS systems detect attacks using software-based feature processing and classifiers. This works, but dense continuous computation is not ideal for low-power always-on monitoring.

Key message:
We explore a spike-based neuromorphic IDS pipeline for hardware-oriented low-power detection.

## Slide 3: System Architecture

Show architecture diagram:
UNSW-NB15 -> preprocessing -> PCA -> spike encoding -> PWL -> LIF -> spike features -> classifier.

Key message:
The PWL waveform is the bridge between Python processing and Cadence/LTSpice circuit simulation.

## Slide 4: Data and Spike Pipeline Proof

Show:
- `processed_data/pca_analog_features.csv`
- 8 PCA neuron columns
- `pwl_sources/sample_0_cat_6/V_neuron_0_m_1.txt`

Say:
We convert real IDS samples into 0 V / 1 V time-voltage source files.

## Slide 5: PWL-Driven LIF Surrogate

Show:
`phase2_results/pwl_lif_surrogate_trace.png`

Say:
This trace uses an actual generated PWL file, not a manually typed waveform.

## Slide 6: Cadence LIF Evidence

Show:
`phase2_results/cadence_lif_waveform_cleaned.png`

Say:
The Cadence waveform shows membrane integration, threshold firing and reset. We corrected spike counting using rising-edge detection.

## Slide 7: Comparison Study

Show:
`phase2_results/phase2_method_comparison.png`

Say:
LIF-derived features preserve IDS accuracy compared to the PCA baseline.

Numbers:
- PCA baseline: 99.15%
- Spike encoded: 99.18%
- LIF derived: 99.20%

## Slide 8: Confusion Matrix and Sparse Activity

Show:
`phase2_results/randomforest_confusion_matrix.png`

Say:
The LIF-derived representation also produces sparse event activity.

Numbers:
- Dense slots per sample: 320
- Average LIF output events: 75.64
- Output sparsity: 76.36%

## Slide 9: Honest Scope and Future Work

Say:
The 99.20% result is from the Python LIF-derived software pipeline, not the full Cadence hardware array. The Cadence LIF currently validates neuron behavior. Next, we will run generated PWL files directly through Cadence and calibrate the surrogate using exported waveforms.

## Slide 10: Final Takeaway

We completed the dataset-to-spike-to-PWL pipeline, validated LIF behavior in Cadence, and showed that LIF-derived spike features preserve IDS performance while moving toward sparse event-driven hardware.

