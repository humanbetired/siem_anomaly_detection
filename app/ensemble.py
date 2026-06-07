import numpy as np
import torch
from app.model import load_models, preprocess


def get_severity(score: float) -> str:
    if score >= 0.85:
        return "CRITICAL"
    elif score >= 0.65:
        return "HIGH"
    elif score >= 0.45:
        return "MEDIUM"
    return "LOW"


def predict(raw_features: dict) -> dict:
    """Ensemble prediction: Isolation Forest + Autoencoder."""
    iso_forest, autoencoder, scaler, features, ae_threshold = load_models()

    # Preprocess
    X = preprocess(raw_features)

    # ── Isolation Forest ──
    if_pred = iso_forest.predict(X)[0]
    if_anomaly = if_pred == -1
    if_score = iso_forest.score_samples(X)[0]
    # Convert ke 0-1 (score negatif, makin negatif makin anomali)
    if_normalized = max(0, min(1, (-if_score + 0.5)))

    # ── Autoencoder ──
    x_tensor = torch.FloatTensor(X)
    with torch.no_grad():
        reconstructed = autoencoder(x_tensor)
        ae_error = float(
            torch.mean((x_tensor - reconstructed) ** 2).item()
        )
    ae_anomaly = ae_error > ae_threshold
    # Normalize error ke 0-1
    ae_normalized = min(1.0, ae_error / (ae_threshold * 2))

    # ── Ensemble decision ──
    # Anomali kalau salah satu atau keduanya flag
    is_anomaly = if_anomaly or ae_anomaly

    # Combined score — weight IF lebih tinggi karena F1 lebih baik
    anomaly_score = (0.55 * if_normalized) + (0.45 * ae_normalized)
    anomaly_score = round(float(anomaly_score), 4)

    severity = get_severity(anomaly_score) if is_anomaly else "LOW"

    return {
        "is_anomaly": bool(is_anomaly),
        "severity": severity,
        "anomaly_score": anomaly_score,
        "isolation_forest_result": "ANOMALY" if if_anomaly else "NORMAL",
        "autoencoder_result": "ANOMALY" if ae_anomaly else "NORMAL",
        "reconstruction_error": round(ae_error, 6),
        "message": "DDoS attack detected" if is_anomaly else "Normal traffic"
    }