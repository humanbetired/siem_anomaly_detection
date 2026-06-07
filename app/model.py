import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import warnings
warnings.filterwarnings('ignore')

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data", "processed")

# ── Dense Autoencoder definition (sama dengan training) ──
class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()

        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── Model loader ──────────────────────────────────────────
_isolation_forest = None
_autoencoder      = None
_scaler           = None
_features         = None
_ae_threshold     = None


def load_models():
    global _isolation_forest, _autoencoder, _scaler, _features, _ae_threshold

    if _isolation_forest is None:
        print("Loading models...")

        # Isolation Forest
        _isolation_forest = joblib.load(
            os.path.join(data_dir, "isolation_forest_best.pkl")
        )

        # Scaler & features
        _scaler   = joblib.load(os.path.join(data_dir, "scaler.pkl"))
        _features = joblib.load(os.path.join(data_dir, "features.pkl"))

        # Autoencoder
        config = joblib.load(os.path.join(data_dir, "autoencoder_config.pkl"))
        input_dim = len(_features)

        ae = DenseAutoencoder(
            input_dim=input_dim,
            hidden_dims=config["hidden_dims"],
            latent_dim=config["latent_dim"]
        )
        ae.load_state_dict(torch.load(
            os.path.join(data_dir, "autoencoder_best.pt"),
            map_location="cpu",
            weights_only=True
        ))
        ae.eval()
        _autoencoder = ae

        # Threshold
        _ae_threshold = joblib.load(
            os.path.join(data_dir, "autoencoder_threshold.pkl")
        )

        print("All models loaded!")

    return _isolation_forest, _autoencoder, _scaler, _features, _ae_threshold


def preprocess(raw_features: dict) -> np.ndarray:
    """Preprocess raw input sama seperti saat training."""
    features = joblib.load(os.path.join(data_dir, "features.pkl"))

    df = pd.DataFrame([raw_features])

    # Engineered features
    df['pkt_len_ratio'] = df.get('Bwd Packet Length Mean', 0) / \
                          (df.get('Fwd Packet Length Mean', 0) + 1)
    df['bwd_fwd_ratio'] = df.get('Bwd Packets/s', 0) / \
                          (df.get('Flow Packets/s', 0) + 1)

    # Pastikan semua kolom ada
    for col in features:
        if col not in df.columns:
            df[col] = 0.0

    X = df[features].values
    scaler = joblib.load(os.path.join(data_dir, "scaler.pkl"))
    X_scaled = scaler.transform(X)
    X_scaled = np.clip(X_scaled, -10, 10)

    return X_scaled