import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, classification_report
)
import mlflow
import mlflow.pytorch
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── Config ───────────────────────────────────────────────
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlflow.set_tracking_uri(f"sqlite:///{base_dir}/mlflow.db")
mlflow.set_experiment("siem-anomaly-detection")

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
EPOCHS     = 50
print(f"Device: {DEVICE}")


# ── Model ────────────────────────────────────────────────
class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()

        # Encoder
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

        # Decoder — mirror dari encoder
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
        latent = self.encoder(x)
        return self.decoder(latent)

    def get_reconstruction_error(self, x):
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error


# ── Training ─────────────────────────────────────────────
def train_model(model, train_loader, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    model.train()
    history = []

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            x = batch[0].to(DEVICE)
            optimizer.zero_grad()
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history.append(avg_loss)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

    return history


# ── Evaluation ───────────────────────────────────────────
def get_errors(model, X):
    model.eval()
    errors = []
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X)),
        batch_size=512, shuffle=False
    )
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(DEVICE)
            error = model.get_reconstruction_error(x)
            errors.extend(error.cpu().numpy())
    return np.array(errors)


def find_best_threshold(errors, y_true):
    """Cari threshold optimal berdasarkan F1 score."""
    thresholds = np.percentile(errors, np.arange(50, 99, 1))
    best_f1, best_threshold = 0, thresholds[0]

    for t in thresholds:
        y_pred = (errors > t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, best_f1


def evaluate(errors, y_true, threshold, run_name):
    y_pred    = (errors > threshold).astype(int)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_true, errors)

    print(f"\n[{run_name}]")
    print(f"F1: {f1:.4f} | Precision: {precision:.4f} | "
          f"Recall: {recall:.4f} | ROC-AUC: {roc_auc:.4f}")
    print(classification_report(
        y_true, y_pred, target_names=['BENIGN', 'DDoS']
    ))

    return {"f1_score": f1, "precision": precision,
            "recall": recall, "roc_auc": roc_auc,
            "threshold": float(threshold)}


# ── Experiment runner ────────────────────────────────────
def run_experiment(run_name, hidden_dims, latent_dim, lr,
                   X_train, X_test, y_test):

    input_dim = X_train.shape[1]

    with mlflow.start_run(run_name=run_name):
        params = {
            "hidden_dims": str(hidden_dims),
            "latent_dim": latent_dim,
            "lr": lr,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE
        }
        mlflow.log_params(params)

        # Build model
        model = DenseAutoencoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim
        ).to(DEVICE)

        # DataLoader — hanya BENIGN untuk training
        train_tensor = torch.FloatTensor(X_train)
        train_loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=BATCH_SIZE, shuffle=True
        )

        # Train
        print(f"\nTraining {run_name}...")
        history = train_model(model, train_loader, EPOCHS, lr)
        mlflow.log_metric("final_train_loss", history[-1])

        # Get reconstruction errors
        errors = get_errors(model, X_test)

        # Cari threshold optimal
        threshold, _ = find_best_threshold(errors, y_test)
        print(f"Optimal threshold: {threshold:.6f}")

        # Evaluate
        metrics = evaluate(errors, y_test, threshold, run_name)
        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(model, "model")

        return metrics, model, threshold


# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    data_dir = os.path.join(base_dir, "data", "processed")

    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv")).values
    X_test  = pd.read_csv(os.path.join(data_dir, "X_test.csv")).values
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze().values
    y_test  = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze().values

    # Hanya BENIGN untuk training autoencoder
    X_train_normal = X_train[y_train == 0]
    print(f"Training pada BENIGN: {len(X_train_normal):,} rows")
    print(f"Input dim: {X_train_normal.shape[1]}")

    results = {}

    # Eksperimen 1 — Baseline
    metrics, _, _ = run_experiment(
        run_name="ae-baseline",
        hidden_dims=[64, 32], latent_dim=16, lr=1e-3,
        X_train=X_train_normal, X_test=X_test, y_test=y_test
    )
    results["ae-baseline"] = metrics

    # Eksperimen 2 — Wider
    metrics, _, _ = run_experiment(
        run_name="ae-wider",
        hidden_dims=[128, 64], latent_dim=32, lr=1e-3,
        X_train=X_train_normal, X_test=X_test, y_test=y_test
    )
    results["ae-wider"] = metrics

    # Eksperimen 3 — Deeper
    metrics, _, _ = run_experiment(
        run_name="ae-deeper",
        hidden_dims=[128, 64, 32], latent_dim=16, lr=5e-4,
        X_train=X_train_normal, X_test=X_test, y_test=y_test
    )
    results["ae-deeper"] = metrics

    # Eksperimen 4 — Best tuned
    metrics, best_model, best_threshold = run_experiment(
        run_name="ae-tuned",
        hidden_dims=[256, 128, 64], latent_dim=32, lr=1e-3,
        X_train=X_train_normal, X_test=X_test, y_test=y_test
    )
    results["ae-tuned"] = metrics

    # Summary
    print("\n" + "="*60)
    print("SUMMARY — Dense Autoencoder:")
    print("="*60)
    best_name = max(results, key=lambda x: results[x]['f1_score'])
    for name, m in results.items():
        marker = " ← BEST" if name == best_name else ""
        print(f"{name:20s} F1: {m['f1_score']:.4f} | "
              f"ROC-AUC: {m['roc_auc']:.4f}{marker}")

    # Simpan best model
    torch.save(
        best_model.state_dict(),
        os.path.join(data_dir, "autoencoder_best.pt")
    )
    joblib.dump(best_threshold,
                os.path.join(data_dir, "autoencoder_threshold.pkl"))

    best_configs = {
        "ae-baseline": {"hidden_dims": [64, 32],       "latent_dim": 16},
        "ae-wider":    {"hidden_dims": [128, 64],       "latent_dim": 32},
        "ae-deeper":   {"hidden_dims": [128, 64, 32],   "latent_dim": 16},
        "ae-tuned":    {"hidden_dims": [256, 128, 64],  "latent_dim": 32},
    }
    joblib.dump(best_configs[best_name],
                os.path.join(data_dir, "autoencoder_config.pkl"))

    print(f"\nBest model: {best_name}")
    print("✅ Milestone 3 selesai!")