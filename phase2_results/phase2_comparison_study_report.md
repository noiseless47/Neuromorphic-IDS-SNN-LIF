# Phase 2 Comparison Study

Generated: 2026-05-07 21:45:22

## Core Claim

The proposed neuromorphic IDS pipeline keeps detection performance close to the conventional PCA-based software baseline while converting the representation into sparse spike and LIF-derived features that are suitable for a Cadence-validated analog neuron.

## Best Result Per Method

| Method | Best model | Accuracy | Weighted F1 | Attack recall | False positive rate | Feature count |
|---|---:|---:|---:|---:|---:|---:|
| PCA_Baseline | RandomForest | 99.15% | 99.15% | 99.80% | 1.50% | 8 |
| Spike_Encoded | RandomForest | 99.18% | 99.18% | 99.83% | 1.47% | 101 |
| LIF_Derived | RandomForest | 99.20% | 99.20% | 99.83% | 1.43% | 168 |

## Efficiency-Style Evidence

- Dense spike-time slots per sample: 320
- Average input spike events per sample after encoding: 168.85
- Average LIF output events per sample: 75.64
- Input event sparsity vs dense time grid: 47.23%
- LIF output sparsity vs dense time grid: 76.36%

## Generated PWL Evidence

- PWL source used: `d:/Neuromorphic-IDS-SNN-LIF/pwl_sources/sample_0_cat_6/V_neuron_0_m_1.txt`
- Input spike ticks in selected PWL: 2
- LIF surrogate output spikes from that PWL: 1
- PWL-driven trace CSV: `d:/Neuromorphic-IDS-SNN-LIF/phase2_results\pwl_lif_surrogate_trace.csv`
- PWL-driven trace plot: `d:/Neuromorphic-IDS-SNN-LIF/phase2_results\pwl_lif_surrogate_trace.png`

## Cadence Evidence

- Cadence notebook waveform was parsed and cleaned.
- Input type: controlled constant 1.2 V input, not IDS PWL
- Corrected rising-edge spike count: 4
- Corrected spike times: [150.0, 200.0, 340.0, 490.0] ns
- Inferred threshold region: 0.52 V
- Inferred reset region: 0.09 V
- Cleaned waveform CSV: `d:/Neuromorphic-IDS-SNN-LIF/phase2_results\cadence_lif_waveform_cleaned.csv`
- Cleaned waveform plot: `d:/Neuromorphic-IDS-SNN-LIF/phase2_results\cadence_lif_waveform_cleaned.png`

## Mentor-Safe Interpretation

The LIF-derived pipeline achieved 99.20% accuracy and 99.20% weighted F1. Compared with the PCA software baseline, the weighted-F1 difference is +0.05 percentage points.

The correct Phase 2 claim is not that the full Cadence array is finished. The stronger and safer claim is that one Cadence LIF validates the physical neuron behavior, while the comparison study shows that spike/LIF-domain features preserve IDS detection quality with sparse event-driven activity.

## Files To Show

- Method comparison plot: `d:/Neuromorphic-IDS-SNN-LIF/phase2_results\phase2_method_comparison.png`
- PWL-driven LIF trace plot: `d:/Neuromorphic-IDS-SNN-LIF/phase2_results\pwl_lif_surrogate_trace.png`
- Full comparison CSV: `d:/Neuromorphic-IDS-SNN-LIF/phase2_results\phase2_comparison_metrics.csv`
- Full comparison JSON: `d:/Neuromorphic-IDS-SNN-LIF/phase2_results\phase2_comparison_metrics.json`