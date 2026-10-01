"""
train_cpu_checkpoint.py
=======================
Pilot-confirmed winner hyperparams, Diabetes split=0, model_seed=0.
No grid search. torch.set_num_threads(8).

Output:
  models_cpu/diabetes_ftt_seed0.pt
  models_cpu/diabetes_mlp_seed0.pt
Each file: {'clf': fitted_estimator, 'state_dict': ..., 'hyperparams': ..., 'verify_pred3': ...}
"""

import os, sys, time, warnings
warnings.filterwarnings('ignore')
os.environ["SEGMENT_DISABLE"] = "1"
os.environ["POSTHOG_DISABLED"] = "1"

import torch
torch.set_num_threads(8)
print(f"torch threads: {torch.get_num_threads()}")

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_data

# Monkey-patch to force CPU (needed during .fit())
import models as _models_module
_orig_FTT = _models_module.FTTransformerClf.__init__
_orig_MLP = _models_module.TorchMLPClf.__init__

def _ftt_cpu(self, num_features, cat_features,
             n_epochs=100, batch_size=256, lr=1e-4, weight_decay=1e-5,
             d_block=192, n_blocks=3, attention_n_heads=8, dropout=0.1,
             device=None, random_state=0):
    _orig_FTT(self, num_features=num_features, cat_features=cat_features,
              n_epochs=n_epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
              d_block=d_block, n_blocks=n_blocks, attention_n_heads=attention_n_heads,
              dropout=dropout, device='cpu', random_state=random_state)

def _mlp_cpu(self, n_epochs=100, batch_size=256, lr=1e-3, weight_decay=1e-4,
             hidden_dims=(128, 64), dropout=0.1, device=None, random_state=0):
    _orig_MLP(self, n_epochs=n_epochs, batch_size=batch_size, lr=lr,
              weight_decay=weight_decay, hidden_dims=hidden_dims, dropout=dropout,
              device='cpu', random_state=random_state)

_models_module.FTTransformerClf.__init__ = _ftt_cpu
_models_module.TorchMLPClf.__init__      = _mlp_cpu
from models import FTTransformerClf, TorchMLPClf

# ── Winner hyperparams (pilot-confirmed) ──────────────────────────────────────
FTT_HP = dict(d_block=128, n_blocks=2, lr=3e-4,
              n_epochs=100, batch_size=256, weight_decay=1e-5,
              attention_n_heads=8, dropout=0.1, random_state=0)
MLP_HP = dict(hidden_dims=(64,), lr=3e-4, weight_decay=1e-4, dropout=0.1,
              n_epochs=100, batch_size=256, random_state=0)

OUT_DIR = 'models_cpu'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load & split ──────────────────────────────────────────────────────────────
X, Y = load_data('diabetes')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for i, (tr, te) in enumerate(skf.split(X, Y)):
    if i == 0: break

X_train = X.iloc[tr].reset_index(drop=True)
Y_train = Y.iloc[tr].reset_index(drop=True)
X_test  = X.iloc[te].reset_index(drop=True)

cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_features = [c for c in X.columns if c not in cat_features]

print(f"train={len(X_train)}, test={len(X_test)}, features={X.columns.tolist()}")

# ── FTT ───────────────────────────────────────────────────────────────────────
print("\n[FTT] Training with winner hyperparams (d_block=128, n_blocks=2, lr=3e-4)...")
t0 = time.time()
clf_ftt = FTTransformerClf(num_features=num_features, cat_features=cat_features, **FTT_HP)
clf_ftt.fit(X_train, Y_train)
ftt_t = time.time() - t0
print(f"  Done in {ftt_t:.1f}s")

ftt_pred3 = clf_ftt.predict_proba(X_test.iloc[:3])[:, 1]
print(f"  Verify test[:3]: {np.round(ftt_pred3, 8).tolist()}")

torch.save({
    'clf':          clf_ftt,
    'state_dict':   clf_ftt.model_.state_dict(),
    'hyperparams':  FTT_HP,
    'num_features': num_features,
    'cat_features': cat_features,
    'verify_pred3': ftt_pred3,
    'dataset': 'diabetes', 'split_seed': 0, 'model_seed': 0,
}, os.path.join(OUT_DIR, 'diabetes_ftt_seed0.pt'))
print(f"  Saved -> {OUT_DIR}/diabetes_ftt_seed0.pt")

# ── MLP ───────────────────────────────────────────────────────────────────────
print("\n[MLP] Training with winner hyperparams (hidden=(64,), lr=3e-4)...")
t0 = time.time()

preprocess = ColumnTransformer(transformers=[
    ("num", Pipeline([("scaler", StandardScaler())]), num_features),
    ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_features),
])
clf_mlp_inner = TorchMLPClf(**MLP_HP)
pipeline_mlp = Pipeline(steps=[('preprocess', preprocess), ('clf', clf_mlp_inner)])
pipeline_mlp.fit(X_train, Y_train)
mlp_t = time.time() - t0
print(f"  Done in {mlp_t:.1f}s")

mlp_pred3 = pipeline_mlp.predict_proba(X_test.iloc[:3])[:, 1]
print(f"  Verify test[:3]: {np.round(mlp_pred3, 8).tolist()}")

torch.save({
    'clf':         pipeline_mlp,
    'state_dict':  pipeline_mlp.named_steps['clf'].model_.state_dict(),
    'hyperparams': MLP_HP,
    'verify_pred3': mlp_pred3,
    'dataset': 'diabetes', 'split_seed': 0, 'model_seed': 0,
}, os.path.join(OUT_DIR, 'diabetes_mlp_seed0.pt'))
print(f"  Saved -> {OUT_DIR}/diabetes_mlp_seed0.pt")

print(f"\nTotal: {ftt_t+mlp_t:.1f}s  |  Checkpoints ready in {OUT_DIR}/")
