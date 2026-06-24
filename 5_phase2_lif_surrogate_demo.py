import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATA_PATH = "d:/Neuromorphic-IDS-SNN-LIF/processed_data/pca_analog_features.csv"
RESULTS_DIR = "d:/Neuromorphic-IDS-SNN-LIF/phase2_results"
DEMO_PWL_PATH = "d:/Neuromorphic-IDS-SNN-LIF/pwl_sources/sample_0_cat_6/V_neuron_0_m_1.txt"

NORMAL_LABEL = None
TIME_STEPS = 10
TIMESTEP_DURATION = 0.001
POPULATION_SIZE = 4


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
    """
    Small discrete-time LIF surrogate for Phase 2 validation.

    This is intentionally simple so it can be calibrated against the one Cadence
    LIF cell: tune tau_mem, threshold, and input_gain until spike times match.
    """
    v_mem = 0.0
    voltages = []
    output_spikes = []
    spike_times = []
    decay = np.exp(-TIMESTEP_DURATION / tau_mem)

    for tick, spike in enumerate(spike_train):
        v_mem = v_mem * decay + resistance * input_gain * spike
        fired = v_mem >= threshold

        if fired:
            output_spikes.append(1.0)
            spike_times.append(tick * TIMESTEP_DURATION)
            voltages.append(v_mem)
            v_mem = reset_voltage
        else:
            output_spikes.append(0.0)
            voltages.append(v_mem)

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


def lif_features_for_row(row, pca_cols):
    features = []
    all_spike_times = []
    all_counts = []
    all_max_voltages = []
    all_auc = []

    for col in pca_cols:
        value = float(row[col])
        activations = population_activations(value)

        for activation in activations:
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

    if all_spike_times:
        global_first = min(all_spike_times)
        global_mean = float(np.mean(all_spike_times))
    else:
        global_first = TIME_STEPS * TIMESTEP_DURATION
        global_mean = TIME_STEPS * TIMESTEP_DURATION

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


def infer_normal_label(df):
    """
    The current pipeline stores only numeric LabelEncoder values, not class names.
    In UNSW-NB15, Normal is the dominant class after stratified sampling, so the
    largest class is the safest reproducible Phase 2 inference.
    """
    if NORMAL_LABEL is not None:
        return NORMAL_LABEL
    return int(df["attack_cat"].value_counts().idxmax())


def build_balanced_binary_dataset(df, normal_label, max_per_class=12000):
    normal_df = df[df["attack_cat"] == normal_label].copy()
    attack_df = df[df["attack_cat"] != normal_label].copy()

    sample_size = min(len(normal_df), len(attack_df), max_per_class)
    normal_df = normal_df.sample(sample_size, random_state=42)
    attack_df = attack_df.sample(sample_size, random_state=42)

    demo_df = pd.concat([normal_df, attack_df], ignore_index=True)
    demo_df = demo_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    y = (demo_df["attack_cat"] != normal_label).astype(int).to_numpy()
    return demo_df, y


def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "SVM_RBF": SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
    }

    results = {}
    predictions = {}
    for name, model in models.items():
        if name == "SVM_RBF":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) else 0.0

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_attack": precision_score(y_test, y_pred, pos_label=1),
            "recall_attack": recall_score(y_test, y_pred, pos_label=1),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "false_positive_rate": fpr,
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                y_test,
                y_pred,
                target_names=["Normal", "Attack"],
                digits=4,
            ),
        }
        predictions[name] = (y_test, y_pred)

    return results, predictions


def export_demo_trace(output_dir):
    input_spikes = load_pwl_ticks(DEMO_PWL_PATH)
    voltages, output_spikes, _ = simulate_lif(input_spikes)

    trace_df = pd.DataFrame(
        {
            "time_s": np.arange(TIME_STEPS) * TIMESTEP_DURATION,
            "input_spike_v": input_spikes,
            "lif_membrane_v": voltages,
            "lif_output_spike": output_spikes,
            "source_pwl": DEMO_PWL_PATH,
        }
    )
    trace_path = os.path.join(output_dir, "single_lif_surrogate_trace.csv")
    trace_df.to_csv(trace_path, index=False)
    return trace_path


def export_plots(results, trace_path, output_dir):
    trace_df = pd.read_csv(trace_path)
    trace_png = os.path.join(output_dir, "single_lif_surrogate_trace.png")

    plt.figure(figsize=(9, 5))
    plt.step(
        trace_df["time_s"] * 1000.0,
        trace_df["input_spike_v"],
        where="post",
        label="Input Spike",
        linewidth=2,
    )
    plt.plot(
        trace_df["time_s"] * 1000.0,
        trace_df["lif_membrane_v"],
        marker="o",
        label="LIF Membrane Voltage",
        linewidth=2,
    )
    plt.step(
        trace_df["time_s"] * 1000.0,
        trace_df["lif_output_spike"],
        where="post",
        label="Output Spike",
        linewidth=2,
    )
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage / Spike State")
    plt.title("Single LIF Surrogate Response")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(trace_png, dpi=180)
    plt.close()

    confusion_paths = {}
    for name, metrics in results.items():
        cm = np.array(metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{name} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1], ["Normal", "Attack"])
        ax.set_yticks([0, 1], ["Normal", "Attack"])

        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(
                    col,
                    row,
                    str(cm[row, col]),
                    ha="center",
                    va="center",
                    color="white" if cm[row, col] > cm.max() * 0.55 else "black",
                    fontsize=12,
                    fontweight="bold",
                )

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = os.path.join(output_dir, f"{name.lower()}_confusion_matrix.png")
        fig.savefig(path, dpi=180)
        plt.close(fig)
        confusion_paths[name] = path

    return trace_png, confusion_paths


def write_report(
    results,
    sample_count,
    feature_count,
    class_counts,
    normal_label,
    trace_path,
    trace_png,
    confusion_paths,
    output_dir,
):
    best_name = max(results, key=lambda name: results[name]["f1_weighted"])
    best = results[best_name]

    report_path = os.path.join(output_dir, "phase2_lif_surrogate_report.md")
    lines = [
        "# Phase 2 LIF Surrogate Validation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Phase 2 Claim",
        "",
        (
            "One Cadence LIF cell has been validated at circuit level. For Phase 2, "
            "the same LIF dynamics are represented as a Python surrogate and reused "
            "across the 8 PCA channels through time-multiplexed evaluation. This shows "
            "that the software-to-spike-to-LIF-to-classifier pipeline is working before "
            "the full 8-LIF parallel array is physically implemented."
        ),
        "",
        "## Dataset Used",
        "",
        f"- Balanced binary IDS samples: {sample_count}",
        f"- LIF-derived features per sample: {feature_count}",
        f"- Inferred Normal label from current dataset: {normal_label}",
        f"- Normal samples: {class_counts.get(0, 0)}",
        f"- Attack samples: {class_counts.get(1, 0)}",
        "",
        "## LIF Surrogate Settings",
        "",
        "- Input voltage spike level: 0 V / 1 V",
        "- Spike timestep: 1 ms",
        "- Window length: 10 ms",
        "- PCA input channels: 8",
        "- Population channels per PCA feature: 4",
        "- Effective LIF evaluations per sample: 32",
        "- Surrogate parameters: tau_mem=6 ms, threshold=1.0 V, input_gain=0.62",
        "",
        "## Classifier Results",
        "",
    ]

    for name, metrics in results.items():
        cm = metrics["confusion_matrix"]
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Accuracy: {metrics['accuracy'] * 100:.2f}%",
                f"- Weighted F1: {metrics['f1_weighted'] * 100:.2f}%",
                f"- Attack precision: {metrics['precision_attack'] * 100:.2f}%",
                f"- Attack recall: {metrics['recall_attack'] * 100:.2f}%",
                f"- False positive rate: {metrics['false_positive_rate'] * 100:.2f}%",
                f"- Confusion matrix [[TN, FP], [FN, TP]]: {cm}",
                f"- Confusion matrix plot: `{confusion_paths[name]}`",
                "",
                "```text",
                metrics["classification_report"],
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Best Phase 2 Talking Point",
            "",
            (
                f"The strongest current model is {best_name}, reaching "
                f"{best['accuracy'] * 100:.2f}% accuracy and "
                f"{best['f1_weighted'] * 100:.2f}% weighted F1 on balanced "
                "Normal-vs-Attack detection using only LIF-derived spike features."
            ),
            "",
            "## What To Show The Mentor",
            "",
            "1. Show the Cadence waveform proving one LIF cell works.",
            "2. Show the PWL input format generated by the spike encoder.",
            "3. Show `single_lif_surrogate_trace.csv` as the Python mirror of the LIF response from a generated PWL file.",
            "4. Show `single_lif_surrogate_trace.png` as the quick visual version.",
            "5. Show this report's accuracy, F1, recall, and false positive rate.",
            "6. Explain that Phase 3 is physical scaling from one verified LIF to an 8-LIF array.",
            "",
            f"Trace file: `{trace_path}`",
            f"Trace plot: `{trace_png}`",
            f"Generated PWL used for trace: `{DEMO_PWL_PATH}`",
            "",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Run 1_data_pipeline.py first.")

    df = pd.read_csv(DATA_PATH)
    pca_cols = [c for c in df.columns if c.lower().startswith("pca_neuron_")]
    if len(pca_cols) != 8:
        raise ValueError(f"Expected 8 PCA neuron columns, found {len(pca_cols)}.")

    normal_label = infer_normal_label(df)
    demo_df, y = build_balanced_binary_dataset(df, normal_label)
    print(f"Building LIF-derived features for {len(demo_df)} balanced samples...")
    X = np.vstack([lif_features_for_row(row, pca_cols) for _, row in demo_df.iterrows()])

    feature_cols = [f"lif_feature_{idx}" for idx in range(X.shape[1])]
    features_df = pd.DataFrame(X, columns=feature_cols)
    features_df["binary_label"] = y
    features_df["original_attack_cat"] = demo_df["attack_cat"].to_numpy()
    features_path = os.path.join(RESULTS_DIR, "lif_surrogate_features.csv")
    features_df.to_csv(features_path, index=False)

    results, predictions = train_models(X, y)

    metrics_path = os.path.join(RESULTS_DIR, "lif_surrogate_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    for model_name, (y_test, y_pred) in predictions.items():
        pred_path = os.path.join(RESULTS_DIR, f"{model_name.lower()}_predictions.csv")
        pd.DataFrame({"actual": y_test, "predicted": y_pred}).to_csv(pred_path, index=False)

    trace_path = export_demo_trace(RESULTS_DIR)
    trace_png, confusion_paths = export_plots(results, trace_path, RESULTS_DIR)
    class_counts = pd.Series(y).value_counts().to_dict()
    report_path = write_report(
        results=results,
        sample_count=len(demo_df),
        feature_count=X.shape[1],
        class_counts=class_counts,
        normal_label=normal_label,
        trace_path=trace_path,
        trace_png=trace_png,
        confusion_paths=confusion_paths,
        output_dir=RESULTS_DIR,
    )

    print("\n=== Phase 2 LIF Surrogate Demo Complete ===")
    print(f"Features: {features_path}")
    print(f"Metrics : {metrics_path}")
    print(f"Trace   : {trace_path}")
    print(f"Plot    : {trace_png}")
    print(f"Report  : {report_path}")
    for name, metrics in results.items():
        print(
            f"{name}: accuracy={metrics['accuracy'] * 100:.2f}%, "
            f"weighted_f1={metrics['f1_weighted'] * 100:.2f}%, "
            f"attack_recall={metrics['recall_attack'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
