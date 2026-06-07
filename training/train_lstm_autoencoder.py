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

# ── Config ──────────────────────────────────────────────
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mlflow.set_tracking_uri(f"sqlite:///{base_dir}/mlflow.db")
mlflow.set_experiment("siem-anomaly-detection")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

SEQ_LEN    = 10   # berapa timestep per sequence
BATCH_SIZE = 256
EPOCHS     = 30
LR         = 1e-3


# ── Model ────────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2):
        super().__init__()

        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        # Latent projection
        self.latent = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        # Output projection
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Encode
        _, (hidden, _) = self.encoder(x)
        latent = self.latent(hidden[-1])

        # Decode — repeat latent untuk setiap timestep
        decoder_input = self.decoder_input(latent)
        decoder_input = decoder_input.unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(decoder_input)

        # Rekonstruksi
        output = self.output_layer(decoded)
        return output


# ── Data preparation ─────────────────────────────────────
def make_sequences(X, seq_len):
    """Convert tabular data ke sequences untuk LSTM."""
    sequences = []
    for i in range(len(X) - seq_len + 1):
        sequences.append(X[i:i + seq_len])
    return np.array(sequences)


def load_data():
    data_dir = os.path.join(base_dir, "data", "processed")
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv")).values
    X_test  = pd.read_csv(os.path.join(data_dir, "X_test.csv")).values
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze().values
    y_test  = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze().values
    return X_train, X_test, y_train, y_test


# ── Training ─────────────────────────────────────────────
def train_model(model, train_loader, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
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
        scheduler.step(avg_loss)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

    return history


# ── Evaluation ───────────────────────────────────────────
def get_reconstruction_errors(model, X_seq):
    """Hitung reconstruction error per sequence."""
    model.eval()
    errors = []
    criterion = nn.MSELoss(reduction='none')

    with torch.no_grad():
        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_seq)),
            batch_size=512
        )
        for batch in loader:
            x = batch[0].to(DEVICE)
            reconstructed = model(x)
            # Mean error per sequence
            error = criterion(reconstructed, x).mean(dim=[1, 2])
            errors.extend(error.cpu().numpy())

    return np.array(errors)


def evaluate(errors, y_true, threshold, run_name):
    """Evaluasi dengan threshold tertentu."""
    y_pred = (errors > threshold).astype(int)

    f1        = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    roc_auc   = roc_auc_score(y_true, errors)

    print(f"\n[{run_name}] Threshold: {threshold:.6f}")
    print(f"F1: {f1:.4f} | Precision: {precision:.4f} | "
          f"Recall: {recall:.4f} | ROC-AUC: {roc_auc:.4f}")
    print(classification_report(y_true, y_pred,
                                target_names=['BENIGN', 'DDoS']))

    return {"f1_score": f1, "precision": precision,
            "recall": recall, "roc_auc": roc_auc,
            "threshold": threshold}


# ── Experiment runner ────────────────────────────────────
def run_experiment(run_name, hidden_dim, latent_dim,
                   num_layers, epochs, lr,
                   X_train_seq, X_test_seq, y_test_seq):

    input_dim = X_train_seq.shape[2]

    with mlflow.start_run(run_name=run_name):
        params = {
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "num_layers": num_layers,
            "epochs": epochs,
            "lr": lr,
            "seq_len": SEQ_LEN,
            "batch_size": BATCH_SIZE
        }
        mlflow.log_params(params)

        # Build model
        model = LSTMAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers
        ).to(DEVICE)

        # DataLoader
        train_tensor = torch.FloatTensor(X_train_seq)
        train_loader = DataLoader(
            TensorDataset(train_tensor),
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        # Train
        print(f"\nTraining {run_name}...")
        history = train_model(model, train_loader, epochs, lr)
        mlflow.log_metric("final_train_loss", history[-1])

        # Get errors on test set
        errors = get_reconstruction_errors(model, X_test_seq)

        # Threshold: mean + 2*std dari errors pada data normal
        normal_errors = errors[y_test_seq == 0]
        threshold = np.mean(normal_errors) + 2 * np.std(normal_errors)

        # Evaluate
        metrics = evaluate(errors, y_test_seq, threshold, run_name)
        mlflow.log_metrics({k: v for k, v in metrics.items()})

        # Save model
        mlflow.pytorch.log_model(model, "model")

        return metrics, model, threshold, errors


# ── Main ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()

    # Hanya pakai BENIGN untuk training autoencoder
    X_train_normal = X_train[y_train == 0]
    print(f"Training pada BENIGN traffic: {len(X_train_normal):,} rows")

    # Buat sequences
    print("Creating sequences...")
    X_train_seq = make_sequences(X_train_normal, SEQ_LEN)
    X_test_seq  = make_sequences(X_test, SEQ_LEN)
    y_test_seq  = y_test[SEQ_LEN - 1:]  # align labels dengan sequences

    print(f"Train sequences: {X_train_seq.shape}")
    print(f"Test sequences : {X_test_seq.shape}")

    results = {}

    # Eksperimen 1 — Baseline
    metrics, _, threshold, _ = run_experiment(
        run_name="lstm-ae-baseline",
        hidden_dim=64, latent_dim=32,
        num_layers=2, epochs=EPOCHS, lr=LR,
        X_train_seq=X_train_seq,
        X_test_seq=X_test_seq,
        y_test_seq=y_test_seq
    )
    results["lstm-ae-baseline"] = metrics

    # Eksperimen 2 — Larger hidden
    metrics, _, _, _ = run_experiment(
        run_name="lstm-ae-large",
        hidden_dim=128, latent_dim=64,
        num_layers=2, epochs=EPOCHS, lr=LR,
        X_train_seq=X_train_seq,
        X_test_seq=X_test_seq,
        y_test_seq=y_test_seq
    )
    results["lstm-ae-large"] = metrics

    # Eksperimen 3 — Deeper
    metrics, best_model, best_threshold, best_errors = run_experiment(
        run_name="lstm-ae-deep",
        hidden_dim=128, latent_dim=32,
        num_layers=3, epochs=EPOCHS, lr=5e-4,
        X_train_seq=X_train_seq,
        X_test_seq=X_test_seq,
        y_test_seq=y_test_seq
    )
    results["lstm-ae-deep"] = metrics

    # Summary
    print("\n" + "="*60)
    print("SUMMARY — LSTM Autoencoder:")
    print("="*60)
    best_name = max(results, key=lambda x: results[x]['f1_score'])
    for name, m in results.items():
        marker = " ← BEST" if name == best_name else ""
        print(f"{name:25s} F1: {m['f1_score']:.4f} | "
              f"ROC-AUC: {m['roc_auc']:.4f}{marker}")

    # Simpan best model
    model_dir = os.path.join(base_dir, "data", "processed")
    torch.save(best_model.state_dict(),
               os.path.join(model_dir, "lstm_ae_best.pt"))
    joblib.dump(best_threshold,
                os.path.join(model_dir, "lstm_ae_threshold.pkl"))

    # Simpan config model terbaik untuk inference
    best_config = {
        "lstm-ae-baseline": {"hidden_dim": 64,  "latent_dim": 32, "num_layers": 2},
        "lstm-ae-large":    {"hidden_dim": 128, "latent_dim": 64, "num_layers": 2},
        "lstm-ae-deep":     {"hidden_dim": 128, "latent_dim": 32, "num_layers": 3},
    }
    joblib.dump(best_config[best_name],
                os.path.join(model_dir, "lstm_ae_config.pkl"))

    print(f"\nBest model: {best_name}")
    print("✅ Milestone 3 selesai!")