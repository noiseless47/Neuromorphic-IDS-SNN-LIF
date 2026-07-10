import pandas as pd
import numpy as np
import os
import time
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# Configure logging to append to a file with timestamps
log_filename = "classifier_execution.log"
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

def run_benchmarks():
    data_path = "processed_data/pca_analog_features.csv"
    if not os.path.exists(data_path):
        logger.error(f"Error: Could not find {data_path}")
        return

    logger.info("Loading dataset...")
    df = pd.read_csv(data_path)
    # Use a realistic sample for benchmark speed
    df = df.sample(min(10000, len(df)), random_state=42)
    
    X = df.drop(columns=['attack_cat']).values
    y = df['attack_cat'].values
    
    # Binary classification for ROC/AUC (Normal = 0, Attack = 1)
    # Assuming the dataset label is categorical or numeric, let's binarize
    # if y has multiple classes, we'll convert 'Normal' (or equivalent 0) vs rest
    if len(np.unique(y)) > 2:
        # Assuming 0 is normal, >0 is attack for this demo
        y_binary = (y > 0).astype(int)
    else:
        y_binary = y

    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting (XGBoost Approx)": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "MLP (Deep Learning)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }

    results = []

    logger.info("\nStarting Benchmarks...\n" + "="*50)
    for name, model in models.items():
        logger.info(f"Training {name}...")
        start_time = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        inference_time = (time.perf_counter() - start_time) / len(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        
        # Power estimation (15W CPU)
        energy_mj = inference_time * 15.0 * 1000.0
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "AUC": auc,
            "Inference_Time_ms": inference_time * 1000.0,
            "Energy_mJ": energy_mj,
            "CM": cm
        })
        
    logger.info("\n" + "="*50 + "\nBENCHMARK RESULTS\n" + "="*50)
    for r in results:
        logger.info(f"Model: {r['Model']}")
        logger.info(f"  Accuracy : {r['Accuracy']*100:.2f}%")
        logger.info(f"  Precision: {r['Precision']:.4f}")
        logger.info(f"  Recall   : {r['Recall']:.4f}")
        logger.info(f"  F1 Score : {r['F1']:.4f}")
        logger.info(f"  ROC AUC  : {r['AUC']:.4f}")
        logger.info(f"  Energy   : {r['Energy_mJ']:.6f} mJ/inference")
        logger.info(f"  Conf Mat : \n{r['CM']}")
        logger.info("-" * 50)

if __name__ == "__main__":
    run_benchmarks()
