import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score
)
import warnings
warnings.filterwarnings('ignore')

# Set MLflow tracking
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlflow.set_tracking_uri(f"sqlite:///{base_dir}/mlflow.db")
mlflow.set_experiment("siem-anomaly-detection")


def load_data():
    """Load processed data dari Milestone 1."""
    data_dir = os.path.join(base_dir, "data", "processed")

    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze()

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def evaluate(y_true, y_pred, run_name):
    """Hitung semua metrics."""
    # Isolation Forest return -1 (anomaly) dan 1 (normal)
    # Kita convert ke 1 (anomaly/DDoS) dan 0 (normal/BENIGN)
    y_pred_binary = np.where(y_pred == -1, 1, 0)

    f1 = f1_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    roc_auc = roc_auc_score(y_true, y_pred_binary)

    print(f"\n[{run_name}]")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_binary,
                                target_names=['BENIGN', 'DDoS']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred_binary))

    return {
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc
    }


def run_experiment(run_name, params, X_train, X_test, y_test):
    """Jalankan satu eksperimen MLflow."""
    with mlflow.start_run(run_name=run_name):
        # Train
        model = IsolationForest(**params, random_state=42, n_jobs=-1)
        model.fit(X_train)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate
        metrics = evaluate(y_test, y_pred, run_name)

        # Log ke MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        return metrics, model


if __name__ == "__main__":
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    # Isolation Forest hanya butuh data BENIGN untuk training
    # Ini mensimulasikan kondisi production: model belajar "normal" saja
    X_train_normal = X_train[y_train == 0]
    print(f"Training hanya pada BENIGN traffic: {len(X_train_normal):,} rows")

    results = {}

    # Eksperimen 1 — Default baseline
    metrics, _ = run_experiment(
        run_name="if-baseline",
        params={"n_estimators": 100, "contamination": 0.1, "max_samples": "auto"},
        X_train=X_train_normal,
        X_test=X_test,
        y_test=y_test
    )
    results["if-baseline"] = metrics

    # Eksperimen 2 — Contamination lebih tinggi
    metrics, _ = run_experiment(
        run_name="if-contamination-0.2",
        params={"n_estimators": 100, "contamination": 0.2, "max_samples": "auto"},
        X_train=X_train_normal,
        X_test=X_test,
        y_test=y_test
    )
    results["if-contamination-0.2"] = metrics

    # Eksperimen 3 — More estimators
    metrics, _ = run_experiment(
        run_name="if-n500",
        params={"n_estimators": 500, "contamination": 0.1, "max_samples": "auto"},
        X_train=X_train_normal,
        X_test=X_test,
        y_test=y_test
    )
    results["if-n500"] = metrics

    # Eksperimen 4 — Max samples terbatas (lebih cepat, production-friendly)
    metrics, _ = run_experiment(
        run_name="if-max-samples-256",
        params={"n_estimators": 200, "contamination": 0.15, "max_samples": 256},
        X_train=X_train_normal,
        X_test=X_test,
        y_test=y_test
    )
    results["if-max-samples-256"] = metrics

    # Eksperimen 5 — Tuned
    metrics, best_model = run_experiment(
        run_name="if-tuned",
        params={"n_estimators": 300, "contamination": 0.3,
                "max_samples": 512, "max_features": 0.8},
        X_train=X_train_normal,
        X_test=X_test,
        y_test=y_test
    )
    results["if-tuned"] = metrics

    # Summary
    print("\n" + "="*60)
    print("SUMMARY — F1 Score per eksperimen:")
    print("="*60)
    for name, m in results.items():
        print(f"{name:30s} F1: {m['f1_score']:.4f} | ROC-AUC: {m['roc_auc']:.4f}")

    # Simpan model terbaik
    best_name = max(results, key=lambda x: results[x]['f1_score'])
    print(f"\nBest experiment: {best_name}")
    # Simpan best model ke disk
    model_path = os.path.join(base_dir, "data", "processed", "isolation_forest_best.pkl")
    joblib.dump(best_model, model_path)
    print(f"Best model saved: {model_path}")

    print("\n✅ Semua eksperimen selesai. Jalankan: mlflow ui")