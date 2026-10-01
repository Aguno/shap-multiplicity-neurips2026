"""
prepare_arrhythmia.py
=====================
Download UCI Arrhythmia dataset and save as pkl.
- 452 rows, 279 features (+ class column)
- '?' → NaN; drop cols ≥50% missing; drop remaining NaN rows
- Target: class 1 → 0 (normal), classes 2-16 → 1 (arrhythmia)
- Binary features ({0,1} unique values): passthrough (no OHE)
- Continuous features: left raw (StandardScaler applied at train time by models.py)
- Save: dataset/arrhythmia_X.pkl, dataset/arrhythmia_Y.pkl
"""

import os
import io
import pickle
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data"
DATA_DIR = "dataset"

# ── 1. Download ──────────────────────────────────────────────────────────────
print("Downloading arrhythmia.data from UCI ...")
with urllib.request.urlopen(URL) as resp:
    raw = resp.read().decode("utf-8")

df = pd.read_csv(io.StringIO(raw), header=None, na_values="?")
print(f"Raw shape: {df.shape}")  # (452, 280)

# ── 2. Split features / target ───────────────────────────────────────────────
X_raw = df.iloc[:, :-1].copy()   # columns 0–278
Y_raw = df.iloc[:, -1].copy()    # column 279 (class 1–16)

X_raw.columns = [f"col_{i}" for i in range(X_raw.shape[1])]

# ── 3. Missing value table (non-zero only) ───────────────────────────────────
miss = X_raw.isnull().sum()
miss_nonzero = miss[miss > 0].sort_values(ascending=False)
print("\nColumns with missing values:")
print(miss_nonzero.to_string())
print(f"\nTotal columns with any missing: {len(miss_nonzero)}")

# ── 4. Drop columns ≥ 50% missing ────────────────────────────────────────────
thresh = 0.5 * len(X_raw)
cols_to_drop = miss[miss >= thresh].index.tolist()
print(f"\nDropping {len(cols_to_drop)} column(s) with ≥50% missing: {cols_to_drop}")
X_raw = X_raw.drop(columns=cols_to_drop)

# ── 5. Drop rows with remaining NaN ──────────────────────────────────────────
n_before = len(X_raw)
mask_keep = X_raw.notnull().all(axis=1)
X_clean = X_raw[mask_keep].reset_index(drop=True)
Y_clean = Y_raw[mask_keep].reset_index(drop=True)
print(f"Dropped {n_before - len(X_clean)} rows with remaining NaN.")
print(f"After cleaning — N={len(X_clean)}, d={X_clean.shape[1]}")

# ── 6. Binary target ─────────────────────────────────────────────────────────
Y = (Y_clean != 1).astype(int)  # class 1 → 0 (normal), 2-16 → 1 (arrhythmia)
pos_rate = Y.mean()
print(f"\nPositive rate (arrhythmia): {pos_rate:.4f} ({Y.sum()}/{len(Y)})")

# ── 7. Feature typing ─────────────────────────────────────────────────────────
X = X_clean.astype(float)
binary_cols = [c for c in X.columns if set(X[c].dropna().unique()).issubset({0.0, 1.0})]
continuous_cols = [c for c in X.columns if c not in binary_cols]
print(f"\nBinary columns ({{0,1}} unique): {len(binary_cols)}")
print(f"Continuous columns: {len(continuous_cols)}")

# ── 8. Train/test split (StratifiedKFold fold 0) ─────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, Y)):
    if fold_idx == 0:
        break

train_size = len(train_idx)
test_size  = len(test_idx)

# ── 9. Save ───────────────────────────────────────────────────────────────────
os.makedirs(DATA_DIR, exist_ok=True)
with open(os.path.join(DATA_DIR, "arrhythmia_X.pkl"), "wb") as f:
    pickle.dump(X, f, protocol=4)
with open(os.path.join(DATA_DIR, "arrhythmia_Y.pkl"), "wb") as f:
    pickle.dump(Y, f, protocol=4)
print("\nSaved: dataset/arrhythmia_X.pkl, dataset/arrhythmia_Y.pkl")

# ── 10. Validation summary ────────────────────────────────────────────────────
print("\n" + "="*52)
print(f"{'Arrhythmia dataset — validation summary':^52}")
print("="*52)
print(f"  {'N (total instances)':<30} {len(X)}")
print(f"  {'d (features after cleaning)':<30} {X.shape[1]}")
print(f"  {'Positive rate':<30} {pos_rate:.4f} (~{pos_rate*100:.1f}%)")
print(f"  {'Train size (fold 0)':<30} {train_size}")
print(f"  {'Test size  (fold 0)':<30} {test_size}")
print(f"  {'Binary columns (passthrough)':<30} {len(binary_cols)}")
print(f"  {'Continuous columns (scaled)':<30} {len(continuous_cols)}")
print("="*52)
