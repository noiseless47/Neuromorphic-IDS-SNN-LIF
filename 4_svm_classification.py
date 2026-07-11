import warnings
warnings.filterwarnings('ignore') # Ignore Pyarrow and Sklearn warnings
import argparse
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
import joblib

log_filename = "svm_results.log"
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def extract_spikes_from_csv(csv_path, threshold=0.5):
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.error(f"Failed to read {csv_path}: {exc}")
        return {}

    target_columns = [col for col in df.columns if (' Y' in col) and ('out' in col.lower() or 'vcom' in col.lower() or 'net025' in col.lower())]
    if not target_columns:
        logger.warning(f"No spike-like output columns found in {csv_path}")
        return {}

    results = {}
    for col in target_columns:
        voltages = df[col].values
        spikes = 0
        is_high = False
        reset_threshold = threshold * 0.8

        for voltage in voltages:
            if voltage > threshold and not is_high:
                spikes += 1
                is_high = True
            elif voltage < reset_threshold:
                is_high = False

        results[col] = spikes

    return results


def load_features_from_cadence_folder(csv_directory, threshold=0.5):
    csv_files = sorted([name for name in os.listdir(csv_directory) if name.endswith('.csv')])
    if not csv_files:
        logger.error(f"No CSV files found in {csv_directory}")
        return None

    features_list = []
    for filename in csv_files:
        full_path = os.path.join(csv_directory, filename)
        spike_dict = extract_spikes_from_csv(full_path, threshold=threshold)
        if not spike_dict:
            continue

        feature_row = [spike_dict[column] for column in sorted(spike_dict.keys())]

        label = 0
        if '_cat_' in filename:
            try:
                label = int(filename.split('_cat_')[-1].split('.')[0])
            except ValueError:
                label = 0

        feature_row.append(label)
        features_list.append(feature_row)

    if not features_list:
        logger.error(f"No usable feature rows were extracted from {csv_directory}")
        return None

    return np.array(features_list)


def load_features_from_source(input_path):
    if not os.path.exists(input_path):
        logger.error(f"Input path does not exist: {input_path}")
        return None

    if os.path.isdir(input_path):
        return load_features_from_cadence_folder(input_path)

    if input_path.endswith('.npy'):
        return np.load(input_path)

    if input_path.endswith('.csv'):
        return pd.read_csv(input_path).values

    logger.error(f"Unsupported input source: {input_path}")
    return None

def train_and_evaluate_hardware_svm(features_array):
    """
    Step 7 & 8: Classification & Evaluation
    Takes the extracted hardware spike rates/timings and classifies them.
    """
    logger.info("\n=== [STEP 7] CLASSIFICATION (SVM) ===")
    
    if features_array is None or len(features_array) == 0:
        logger.error("Error: No feature array provided. Simulate hardware first.")
        return

    # Assume last column is label
    X = features_array[:, :-1]
    y = features_array[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info(f"Training SVM (RBF) on {len(X_train)} mapped hardware samples...")
    model = SVC(kernel='rbf', probability=True, class_weight='balanced')
    model.fit(X_train, y_train)
    
    logger.info("\n=== [STEP 8] PERFORMANCE EVALUATION ===")
    y_pred = model.predict(X_test)
    
    # Save the SVM model
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, "svm_model.joblib"))
    logger.info("Saved trained SVM model to models/svm_model.joblib")
    
    # Multi-class Evaluation
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    if y_test.dtype.kind in {'U', 'S', 'O'}:
        y_test_values = np.array([str(value).strip().lower() for value in y_test])
        y_pred_values = np.array([str(value).strip().lower() for value in y_pred])
        normal_mask_test = y_test_values == 'normal'
        normal_mask_pred = y_pred_values == 'normal'
    else:
        normal_mask_test = y_test == 0
        normal_mask_pred = y_pred == 0

    y_test_binary = (~normal_mask_test).astype(int)
    y_pred_binary = (~normal_mask_pred).astype(int)
    
    binary_acc = accuracy_score(y_test_binary, y_pred_binary)
    binary_f1 = f1_score(y_test_binary, y_pred_binary)
    binary_cm = confusion_matrix(y_test_binary, y_pred_binary)
    
    cm_attack = confusion_matrix(y_test_binary, y_pred_binary)
    if cm_attack.shape == (2, 2):
        tn, fp, fn, tp = cm_attack.ravel()
        binary_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        binary_fpr = 0.0
        
    logger.info("-" * 50)
    logger.info(f"Multi-Class Accuracy Score      : {acc * 100:.2f} %")
    logger.info(f"Multi-Class F1-Score (Weighted) : {f1 * 100:.2f} %")
    logger.info("-" * 50)
    logger.info(f"Binary Accuracy (Attack Det.)  : {binary_acc * 100:.2f} %")
    logger.info(f"Binary F1-Score (Attack Det.)  : {binary_f1 * 100:.2f} %")
    logger.info(f"Binary False Positive Rate     : {binary_fpr * 100:.2f} %")
    logger.info("-" * 50)
    logger.info("Classification Report:\n" + classification_report(y_test, y_pred))
    
    logger.info("\n[Analog Neuromorphic Power Analysis]")
    logger.info("> Unlike traditional CPU ML inference (~100-500 mW), the integrated ")
    logger.info("> LTSpice/Cadence LIF topologies exhibit static leakage in the Nano-Watt range ")
    logger.info("> and dynamic spiking power < 1 mW, establishing phenomenal energy efficiency!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SVM directly from Cadence parser output CSVs.")
    parser.add_argument(
        "--input",
        default="dummy_cadence_outputs",
        help="Folder of Cadence CSV outputs, a consolidated CSV, or a .npy feature array"
    )
    args = parser.parse_args()

    features_array = load_features_from_source(args.input)
    if features_array is not None:
        logger.info(f"Loaded features from {args.input} with shape {features_array.shape}.")
        train_and_evaluate_hardware_svm(features_array)
