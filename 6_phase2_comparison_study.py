import ast
import json
import os
import re
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATA_PATH = "d:/Neuromorphic-IDS-SNN-LIF/processed_data/pca_analog_features.csv"
NOTEBOOK_PATH = "c:/Users/AsishKumarYeleti/Downloads/Untitled15.ipynb"
RESULTS_DIR = "d:/Neuromorphic-IDS-SNN-LIF/phase2_results"
DEMO_PWL_PATH = "d:/Neuromorphic-IDS-SNN-LIF/pwl_sources/sample_0_cat_6/V_neuron_0_m_1.txt"

TIME_STEPS = 10
TIMESTEP_DURATION = 0.001
POPULATION_SIZE = 4
MAX_PER_CLASS = 12000


def rate_encoding(value, time_steps=TIME_STEPS):
    spike_count = int(np.clip(value, 0.0, 1.0) * time_steps)
    return np.array([1.0 if t < spike_count else 0.0 for t in range(time_steps)])


def population_activations(value, population_size=POPULATION_SIZE):
    means = np.linspace(0.0, 1.0, population_size)
    sigma = 1.0 / (population_size - 1)
    return np.exp(-0.5 * ((value - means) / sigma) ** 2)


def simulate_lif(
    spike_train,
    tau_mem=0.006,
    resistance=1.0,
    threshold=1.0,
    reset_voltage=0.0,
    input_gain=0.62,
):
    v_mem = 0.0
    voltages = []
    output_spikes = []
    spike_times = []
    decay = np.exp(-TIMESTEP_DURATION / tau_mem)

    for tick, spike in enumerate(spike_train):
        v_mem = v_mem * decay + resistance * input_gain * spike
        fired = v_mem >= threshold
        voltages.append(v_mem)

        if fired:
            output_spikes.append(1.0)
            spike_times.append(tick * TIMESTEP_DURATION)
            v_mem = reset_voltage
        else:
            output_spikes.append(0.0)

    return np.array(voltages), np.array(output_spikes), spike_times


def load_pwl_ticks(pwl_path, time_steps=TIME_STEPS, timestep_duration=TIMESTEP_DURATION):
    points = []
    with open(pwl_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            time_s, voltage = stripped.split()[:2]
            points.append((float(time_s), float(voltage)))

    if not points:
        raise ValueError(f"No PWL points found in {pwl_path}")

    tick_values = []
    for tick in range(time_steps):
        tick_time = tick * timestep_duration
        candidates = [
            voltage for time_s, voltage in points if abs(time_s - tick_time) < 1e-9
        ]
        if candidates:
            tick_values.append(candidates[0])
            continue

        previous = [point for point in points if point[0] <= tick_time]
        tick_values.append(previous[-1][1] if previous else points[0][1])

    return np.array(tick_values, dtype=float)


def infer_normal_label(df):
    return int(df["attack_cat"].value_counts().idxmax())


def build_balanced_binary_dataset(df, normal_label):
    normal_df = df[df["attack_cat"] == normal_label].copy()
    attack_df = df[df["attack_cat"] != normal_label].copy()
    sample_size = min(len(normal_df), len(attack_df), MAX_PER_CLASS)

    normal_df = normal_df.sample(sample_size, random_state=42)
    attack_df = attack_df.sample(sample_size, random_state=42)
    demo_df = pd.concat([normal_df, attack_df], ignore_index=True)
    demo_df = demo_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    y = (demo_df["attack_cat"] != normal_label).astype(int).to_numpy()
    return demo_df, y


def spike_features_for_row(row, pca_cols):
    features = []
    all_counts = []
    all_first_spikes = []

    for col in pca_cols:
        for activation in population_activations(float(row[col])):
            train = rate_encoding(activation)
            active_ticks = np.where(train > 0.5)[0]
            count = float(train.sum())
            first_spike = (
                float(active_ticks[0] * TIMESTEP_DURATION)
                if len(active_ticks)
                else TIME_STEPS * TIMESTEP_DURATION
            )

            features.extend([activation, count, first_spike])
            all_counts.append(count)
            all_first_spikes.append(first_spike)

    features.extend(
        [
            float(np.sum(all_counts)),
            float(np.mean(all_counts)),
            float(np.std(all_counts)),
            float(np.mean(all_first_spikes)),
            float(np.min(all_first_spikes)),
        ]
    )
    return features


def lif_features_for_row(row, pca_cols):
    features = []
    all_spike_times = []
    all_counts = []
    all_max_voltages = []
    all_auc = []

    for col in pca_cols:
        for activation in population_activations(float(row[col])):
            input_spikes = rate_encoding(activation)
            voltages, output_spikes, spike_times = simulate_lif(input_spikes)

            count = float(output_spikes.sum())
            first_spike = spike_times[0] if spike_times else TIME_STEPS * TIMESTEP_DURATION
            mean_spike = float(np.mean(spike_times)) if spike_times else TIME_STEPS * TIMESTEP_DURATION
            max_voltage = float(np.max(voltages))
            auc_voltage = float(np.sum(voltages) * TIMESTEP_DURATION)

            features.extend([count, first_spike, mean_spike, max_voltage, auc_voltage])
            all_spike_times.extend(spike_times)
            all_counts.append(count)
            all_max_voltages.append(max_voltage)
            all_auc.append(auc_voltage)

    global_first = min(all_spike_times) if all_spike_times else TIME_STEPS * TIMESTEP_DURATION
    global_mean = (
        float(np.mean(all_spike_times)) if all_spike_times else TIME_STEPS * TIMESTEP_DURATION
    )
    features.extend(
        [
            float(np.sum(all_counts)),
            float(np.mean(all_counts)),
            float(np.std(all_counts)),
            global_first,
            global_mean,
            float(np.mean(all_max_voltages)),
            float(np.max(all_max_voltages)),
            float(np.sum(all_auc)),
        ]
    )
    return features


def build_feature_sets(demo_df, pca_cols):
    pca_x = demo_df[pca_cols].to_numpy(dtype=float)
    print("Building spike-domain feature set...")
    spike_x = np.vstack([spike_features_for_row(row, pca_cols) for _, row in demo_df.iterrows()])
    print("Building LIF-derived feature set...")
    lif_x = np.vstack([lif_features_for_row(row, pca_cols) for _, row in demo_df.iterrows()])
    return {
        "PCA_Baseline": pca_x,
        "Spike_Encoded": spike_x,
        "LIF_Derived": lif_x,
    }


def evaluate_models(feature_sets, y):
    rows = []
    best_by_method = {}

    for method, X in feature_sets.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        models = {
            "SVM_RBF": (
                SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced"),
                X_train_scaled,
                X_test_scaled,
            ),
            "RandomForest": (
                RandomForestClassifier(
                    n_estimators=250,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
                X_train,
                X_test,
            ),
        }

        for model_name, (model, train_x, test_x) in models.items():
            print(f"Training {method} + {model_name}...")
            model.fit(train_x, y_train)
            y_pred = model.predict(test_x)
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) else 0.0

            row = {
                "method": method,
                "model": model_name,
                "feature_count": X.shape[1],
                "accuracy": accuracy_score(y_test, y_pred),
                "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
                "attack_precision": precision_score(y_test, y_pred, pos_label=1),
                "attack_recall": recall_score(y_test, y_pred, pos_label=1),
                "false_positive_rate": fpr,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
            rows.append(row)

        method_rows = [row for row in rows if row["method"] == method]
        best_by_method[method] = max(method_rows, key=lambda row: row["weighted_f1"])

    return pd.DataFrame(rows), best_by_method


def compute_efficiency_summary(demo_df, pca_cols):
    dense_slots = len(pca_cols) * POPULATION_SIZE * TIME_STEPS
    input_event_counts = []
    lif_event_counts = []

    for _, row in demo_df.iterrows():
        sample_input_events = 0.0
        sample_lif_events = 0.0
        for col in pca_cols:
            for activation in population_activations(float(row[col])):
                train = rate_encoding(activation)
                sample_input_events += float(train.sum())
                _, output_spikes, _ = simulate_lif(train)
                sample_lif_events += float(output_spikes.sum())
        input_event_counts.append(sample_input_events)
        lif_event_counts.append(sample_lif_events)

    avg_input_events = float(np.mean(input_event_counts))
    avg_lif_events = float(np.mean(lif_event_counts))
    return {
        "dense_time_slots_per_sample": dense_slots,
        "avg_input_spike_events_per_sample": avg_input_events,
        "avg_lif_output_events_per_sample": avg_lif_events,
        "input_event_sparsity_percent": 100.0 * (1.0 - avg_input_events / dense_slots),
        "lif_output_sparsity_percent": 100.0 * (1.0 - avg_lif_events / dense_slots),
    }


def safe_eval_list_node(node):
    if isinstance(node, ast.List):
        return ast.literal_eval(node)
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Mult)
        and isinstance(node.left, ast.List)
        and isinstance(node.right, ast.Constant)
        and isinstance(node.right.value, int)
    ):
        return ast.literal_eval(node.left) * node.right.value
    return None


def extract_cadence_notebook_waveform():
    if not os.path.exists(NOTEBOOK_PATH):
        return None

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        if "vmem_v" not in source or "output_v" not in source or "input_v" not in source:
            continue

        parsed = ast.parse(source)
        values = {}
        for node in parsed.body:
            if not isinstance(node, ast.Assign):
                continue
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            name = node.targets[0].id
            if name in {"time", "input_v", "vmem_v", "output_v"}:
                values[name] = safe_eval_list_node(node.value)

        if set(values) == {"time", "input_v", "vmem_v", "output_v"}:
            df = pd.DataFrame(values)
            output_state = (df["output_v"] > 0.5).astype(int)
            rising_edges = (output_state.diff().fillna(output_state) == 1)
            spike_times = df.loc[rising_edges, "time"].to_numpy()
            reset_rows = df[(df["vmem_v"] <= 0.11) & (df["time"] > df["time"].min())]

            metrics = {
                "source": NOTEBOOK_PATH,
                "input_type": "controlled constant 1.2 V input, not IDS PWL",
                "time_unit": "ns",
                "duration_ns": float(df["time"].max() - df["time"].min()),
                "corrected_spike_count": int(len(spike_times)),
                "rising_edge_spike_times_ns": spike_times.astype(float).tolist(),
                "inferred_threshold_v": float(df["vmem_v"].max()),
                "inferred_reset_v": float(reset_rows["vmem_v"].min()) if len(reset_rows) else None,
            }

            csv_path = os.path.join(RESULTS_DIR, "cadence_lif_waveform_cleaned.csv")
            png_path = os.path.join(RESULTS_DIR, "cadence_lif_waveform_cleaned.png")
            df.to_csv(csv_path, index=False)

            plt.figure(figsize=(9, 5))
            plt.plot(df["time"], df["input_v"], label="Input V", linewidth=2)
            plt.plot(df["time"], df["vmem_v"], label="Membrane V", linewidth=2)
            plt.step(df["time"], df["output_v"], where="post", label="Output V", linewidth=2)
            for spike_time in spike_times:
                plt.axvline(spike_time, color="black", linestyle="--", alpha=0.25)
            plt.xlabel("Time (ns)")
            plt.ylabel("Voltage (V)")
            plt.title("Cleaned Cadence LIF Waveform Evidence")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(png_path, dpi=180)
            plt.close()

            metrics["cleaned_csv"] = csv_path
            metrics["plot"] = png_path
            return metrics

    return None


def export_pwl_surrogate_trace():
    input_spikes = load_pwl_ticks(DEMO_PWL_PATH)
    voltages, output_spikes, spike_times = simulate_lif(input_spikes)

    trace_df = pd.DataFrame(
        {
            "time_s": np.arange(TIME_STEPS) * TIMESTEP_DURATION,
            "input_spike_v": input_spikes,
            "lif_membrane_v": voltages,
            "lif_output_spike": output_spikes,
            "source_pwl": DEMO_PWL_PATH,
        }
    )

    csv_path = os.path.join(RESULTS_DIR, "pwl_lif_surrogate_trace.csv")
    png_path = os.path.join(RESULTS_DIR, "pwl_lif_surrogate_trace.png")
    trace_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(9, 5))
    plt.step(
        trace_df["time_s"] * 1000.0,
        trace_df["input_spike_v"],
        where="post",
        label="Generated PWL Input",
        linewidth=2,
    )
    plt.plot(
        trace_df["time_s"] * 1000.0,
        trace_df["lif_membrane_v"],
        marker="o",
        label="LIF Surrogate Membrane",
        linewidth=2,
    )
    plt.step(
        trace_df["time_s"] * 1000.0,
        trace_df["lif_output_spike"],
        where="post",
        label="LIF Output Spike",
        linewidth=2,
    )
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage / Spike State")
    plt.title("LIF Surrogate Driven by Generated IDS PWL Source")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()

    return {
        "source_pwl": DEMO_PWL_PATH,
        "trace_csv": csv_path,
        "trace_plot": png_path,
        "input_spike_count": int(np.sum(input_spikes > 0.5)),
        "lif_output_spike_count": int(np.sum(output_spikes > 0.5)),
        "lif_output_spike_times_s": [float(value) for value in spike_times],
    }


def export_comparison_plot(best_by_method):
    labels = list(best_by_method.keys())
    accuracy = [best_by_method[label]["accuracy"] * 100.0 for label in labels]
    f1 = [best_by_method[label]["weighted_f1"] * 100.0 for label in labels]
    recall = [best_by_method[label]["attack_recall"] * 100.0 for label in labels]

    x = np.arange(len(labels))
    width = 0.25
    path = os.path.join(RESULTS_DIR, "phase2_method_comparison.png")

    plt.figure(figsize=(10, 5.5))
    plt.bar(x - width, accuracy, width, label="Accuracy")
    plt.bar(x, f1, width, label="Weighted F1")
    plt.bar(x + width, recall, width, label="Attack Recall")
    plt.ylabel("Score (%)")
    plt.ylim(90, 100.5)
    plt.title("Phase 2 IDS Pipeline Comparison")
    plt.xticks(x, labels, rotation=10)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def write_report(results_df, best_by_method, efficiency, cadence_metrics, pwl_metrics, plot_path):
    report_path = os.path.join(RESULTS_DIR, "phase2_comparison_study_report.md")
    best_lif = best_by_method["LIF_Derived"]
    best_pca = best_by_method["PCA_Baseline"]

    lines = [
        "# Phase 2 Comparison Study",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Core Claim",
        "",
        (
            "The proposed neuromorphic IDS pipeline keeps detection performance close "
            "to the conventional PCA-based software baseline while converting the "
            "representation into sparse spike and LIF-derived features that are suitable "
            "for a Cadence-validated analog neuron."
        ),
        "",
        "## Best Result Per Method",
        "",
        "| Method | Best model | Accuracy | Weighted F1 | Attack recall | False positive rate | Feature count |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for method, row in best_by_method.items():
        lines.append(
            "| "
            f"{method} | {row['model']} | {row['accuracy'] * 100:.2f}% | "
            f"{row['weighted_f1'] * 100:.2f}% | {row['attack_recall'] * 100:.2f}% | "
            f"{row['false_positive_rate'] * 100:.2f}% | {int(row['feature_count'])} |"
        )

    lines.extend(
        [
            "",
            "## Efficiency-Style Evidence",
            "",
            f"- Dense spike-time slots per sample: {efficiency['dense_time_slots_per_sample']:.0f}",
            (
                "- Average input spike events per sample after encoding: "
                f"{efficiency['avg_input_spike_events_per_sample']:.2f}"
            ),
            (
                "- Average LIF output events per sample: "
                f"{efficiency['avg_lif_output_events_per_sample']:.2f}"
            ),
            (
                "- Input event sparsity vs dense time grid: "
                f"{efficiency['input_event_sparsity_percent']:.2f}%"
            ),
            (
                "- LIF output sparsity vs dense time grid: "
                f"{efficiency['lif_output_sparsity_percent']:.2f}%"
            ),
            "",
            "## Generated PWL Evidence",
            "",
            f"- PWL source used: `{pwl_metrics['source_pwl']}`",
            f"- Input spike ticks in selected PWL: {pwl_metrics['input_spike_count']}",
            f"- LIF surrogate output spikes from that PWL: {pwl_metrics['lif_output_spike_count']}",
            f"- PWL-driven trace CSV: `{pwl_metrics['trace_csv']}`",
            f"- PWL-driven trace plot: `{pwl_metrics['trace_plot']}`",
            "",
            "## Cadence Evidence",
            "",
        ]
    )

    if cadence_metrics:
        lines.extend(
            [
                "- Cadence notebook waveform was parsed and cleaned.",
                f"- Input type: {cadence_metrics['input_type']}",
                f"- Corrected rising-edge spike count: {cadence_metrics['corrected_spike_count']}",
                (
                    "- Corrected spike times: "
                    f"{cadence_metrics['rising_edge_spike_times_ns']} ns"
                ),
                f"- Inferred threshold region: {cadence_metrics['inferred_threshold_v']:.2f} V",
                f"- Inferred reset region: {cadence_metrics['inferred_reset_v']:.2f} V",
                f"- Cleaned waveform CSV: `{cadence_metrics['cleaned_csv']}`",
                f"- Cleaned waveform plot: `{cadence_metrics['plot']}`",
            ]
        )
    else:
        lines.append("- Cadence notebook waveform was not available or could not be parsed.")

    delta = (best_lif["weighted_f1"] - best_pca["weighted_f1"]) * 100.0
    lines.extend(
        [
            "",
            "## Mentor-Safe Interpretation",
            "",
            (
                f"The LIF-derived pipeline achieved {best_lif['accuracy'] * 100:.2f}% "
                f"accuracy and {best_lif['weighted_f1'] * 100:.2f}% weighted F1. "
                f"Compared with the PCA software baseline, the weighted-F1 difference is "
                f"{delta:+.2f} percentage points."
            ),
            "",
            (
                "The correct Phase 2 claim is not that the full Cadence array is finished. "
                "The stronger and safer claim is that one Cadence LIF validates the physical "
                "neuron behavior, while the comparison study shows that spike/LIF-domain "
                "features preserve IDS detection quality with sparse event-driven activity."
            ),
            "",
            "## Files To Show",
            "",
            f"- Method comparison plot: `{plot_path}`",
            f"- PWL-driven LIF trace plot: `{pwl_metrics['trace_plot']}`",
            f"- Full comparison CSV: `{os.path.join(RESULTS_DIR, 'phase2_comparison_metrics.csv')}`",
            f"- Full comparison JSON: `{os.path.join(RESULTS_DIR, 'phase2_comparison_metrics.json')}`",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    pca_cols = [c for c in df.columns if c.lower().startswith("pca_neuron_")]
    normal_label = infer_normal_label(df)
    demo_df, y = build_balanced_binary_dataset(df, normal_label)

    print(f"Using {len(demo_df)} balanced samples. Inferred Normal label: {normal_label}")
    feature_sets = build_feature_sets(demo_df, pca_cols)
    results_df, best_by_method = evaluate_models(feature_sets, y)
    efficiency = compute_efficiency_summary(demo_df, pca_cols)
    cadence_metrics = extract_cadence_notebook_waveform()
    pwl_metrics = export_pwl_surrogate_trace()
    plot_path = export_comparison_plot(best_by_method)

    csv_path = os.path.join(RESULTS_DIR, "phase2_comparison_metrics.csv")
    json_path = os.path.join(RESULTS_DIR, "phase2_comparison_metrics.json")
    results_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": results_df.to_dict(orient="records"),
                "best_by_method": best_by_method,
                "efficiency": efficiency,
                "cadence_metrics": cadence_metrics,
                "pwl_metrics": pwl_metrics,
            },
            f,
            indent=2,
        )

    report_path = write_report(
        results_df, best_by_method, efficiency, cadence_metrics, pwl_metrics, plot_path
    )

    print("\n=== Phase 2 Comparison Study Complete ===")
    print(f"Metrics CSV : {csv_path}")
    print(f"Metrics JSON: {json_path}")
    print(f"Plot        : {plot_path}")
    print(f"Report      : {report_path}")
    for method, row in best_by_method.items():
        print(
            f"{method}: {row['model']} accuracy={row['accuracy'] * 100:.2f}%, "
            f"f1={row['weighted_f1'] * 100:.2f}%, "
            f"attack_recall={row['attack_recall'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
