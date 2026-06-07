# training/rebuild_preprocessing.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data", "processed")

# Load raw data
print("Loading raw data...")
df = pd.read_csv(os.path.join(base_dir, "data",
     "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"))
df.columns = df.columns.str.strip()

# Clean
df = df.drop_duplicates()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()
df['label_encoded'] = (df['Label'] == 'DDoS').astype(int)

# Feature engineering
df['pkt_len_ratio'] = df['Bwd Packet Length Mean'] / \
                      (df['Fwd Packet Length Mean'] + 1)
df['bwd_fwd_ratio'] = df['Bwd Packets/s'] / (df['Flow Packets/s'] + 1)

# Hitung korelasi ulang
df['label_encoded'] = (df['Label'] == 'DDoS').astype(int)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('label_encoded')

correlations = df[numeric_cols].corrwith(df['label_encoded']).abs().sort_values(ascending=False)
top20 = correlations.head(20).index.tolist()

# Final features: top20 + 3 engineered = 23
engineered = ['pkt_len_ratio', 'bwd_fwd_ratio', 'Bwd Packets/s']
features = top20 + engineered

print(f"Total features: {len(features)}")
print(features)

# Load feature list dari milestone 1
features = joblib.load(os.path.join(data_dir, "features.pkl"))

# Tambah engineered features kalau belum ada
for f in ['pkt_len_ratio', 'bwd_fwd_ratio']:
    if f not in features:
        features.append(f)

# Remove flow_rate — ini yang generate inf tadi
features = [f for f in features if f != 'flow_rate']

X = df[features].copy()
y = df['label_encoded'].copy()

# Replace inf dari engineered features
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.dropna()
y = y[X.index]

print(f"Total rows: {len(X):,}")

# Split DULU sebelum scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit scaler HANYA pada BENIGN training data
X_train_benign = X_train[y_train == 0]
print(f"Fitting scaler pada {len(X_train_benign):,} BENIGN rows...")

scaler = RobustScaler()
scaler.fit(X_train_benign)

# Transform semua data dengan scaler yang fit pada BENIGN
X_train_scaled = pd.DataFrame(
    scaler.transform(X_train), columns=features
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test), columns=features
)

# Clip SETELAH scaling — lebih aman
X_train_scaled = X_train_scaled.clip(-10, 10)
X_test_scaled = X_test_scaled.clip(-10, 10)

X_train_scaled = X_train_scaled.reset_index(drop=True)
X_test_scaled = X_test_scaled.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Verifikasi
X_test_benign = X_test_scaled[y_test == 0]
X_test_ddos = X_test_scaled[y_test == 1]

print(f"\n=== Verifikasi setelah fix ===")
print(f"Test BENIGN mean: {X_test_benign.mean().mean():.4f}")
print(f"Test DDoS mean  : {X_test_ddos.mean().mean():.4f}")
print(f"\nTop 3 features:")
for col in features[:3]:
    b = X_test_benign[col].mean()
    d = X_test_ddos[col].mean()
    print(f"  {col:35s} BENIGN: {b:.4f} | DDoS: {d:.4f}")

# Simpan ulang
X_train_scaled.to_csv(os.path.join(data_dir, "X_train.csv"), index=False)
X_test_scaled.to_csv(os.path.join(data_dir, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(data_dir, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(data_dir, "y_test.csv"), index=False)
joblib.dump(scaler, os.path.join(data_dir, "scaler.pkl"))
joblib.dump(features, os.path.join(data_dir, "features.pkl"))

print("\nData rebuilt dan saved!")