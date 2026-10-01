"""
experiment_conditional.py
=========================
Conditional SHAP (ctree) for Diabetes × {FTT, MLP} × 5 seeds.
Loads pre-trained checkpoints from models_cpu/ — no grid search.

--full79 : use all 79 test instances, _full79 output suffix
(default) : stratified 20-instance subsample, no suffix
"""

import os, sys, pickle, time, warnings
warnings.filterwarnings('ignore')
os.environ["SEGMENT_DISABLE"] = "1"
os.environ["POSTHOG_DISABLED"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "8")

import torch
torch.set_num_threads(8)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_data

DATASET    = 'diabetes'
MODELS_RUN = ['ftt', 'mlp']
SEEDS      = [0, 42, 84, 126, 168]
SPLIT_SEED = 0
APPROACH   = 'ctree'
N_MC       = 1000
N_EXPLAIN  = 20
OUT_DIR    = 'results_conditional'
CKPT_DIR   = 'models_cpu'
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load & split ──────────────────────────────────────────────────────────
X, Y = load_data(DATASET)
skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for i, (tr, te) in enumerate(skf.split(X, Y)):
    if i == SPLIT_SEED: break

X_train = X.iloc[tr].reset_index(drop=True)
Y_train = Y.iloc[tr].reset_index(drop=True)
X_test_full = X.iloc[te].reset_index(drop=True)
Y_test_full = Y.iloc[te].reset_index(drop=True)

rng = np.random.RandomState(42)
pos_idx = np.where(Y_test_full.values == 1)[0]
neg_idx = np.where(Y_test_full.values == 0)[0]
n_pos = int(round(N_EXPLAIN * len(pos_idx) / len(Y_test_full)))
n_neg = N_EXPLAIN - n_pos
chosen = np.sort(np.concatenate([
    rng.choice(pos_idx, n_pos, replace=False),
    rng.choice(neg_idx, n_neg, replace=False),
]))
X_test_sub = X_test_full.iloc[chosen].reset_index(drop=True)

print(f"Diabetes split=0: train={len(X_train)}, test_full={len(X_test_full)}, test_sub={len(X_test_sub)}")
print(f"torch threads: {torch.get_num_threads()}", flush=True)

# ── 2. Checkpoint loader ─────────────────────────────────────────────────────
def load_checkpoint(model_name):
    path = os.path.join(CKPT_DIR, f'{DATASET}_{model_name}_seed0.pt')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    return ckpt['clf'], ckpt['verify_pred3']

# ── 3. Predict function for pyshapr ─────────────────────────────────────────
def make_predict_fn(clf):
    def predict_model(model, newdata):
        return model.predict_proba(newdata)[:, 1].astype(float)
    return predict_model

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['ftt', 'mlp', 'both'], default='both')
    parser.add_argument('--full79', action='store_true',
                        help='Use full 79-instance test fold. Output: _full79 suffix.')
    args = parser.parse_args()

    models_to_run = MODELS_RUN if args.model == 'both' else [args.model]
    X_test      = X_test_full.copy() if args.full79 else X_test_sub.copy()
    FILE_SUFFIX = '_full79' if args.full79 else ''
    print(f"[{'full79' if args.full79 else 'sub20'} mode] N_explain={len(X_test)}", flush=True)

    import pyshapr

    # ── Load & verify checkpoints ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("CHECKPOINT VERIFICATION")
    print("="*60)
    clfs = {}
    for model_name in models_to_run:
        clf, saved_pred3 = load_checkpoint(model_name)
        live_pred3 = clf.predict_proba(X_test_full.iloc[:3])[:, 1]
        match = np.array_equal(live_pred3, saved_pred3)
        print(f"  [{model_name.upper()}] test[:3]: {np.round(live_pred3, 8).tolist()}")
        print(f"         vs saved:    {np.round(saved_pred3, 8).tolist()}")
        print(f"         byte-match:  {'PASS ✓' if match else 'FAIL ✗'}", flush=True)
        clfs[model_name] = clf

    # ── Full experiment ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FULL EXPERIMENT")
    print("="*60)

    for model_name in models_to_run:
        print(f"\n{'='*50}\nModel: {model_name.upper()}\n{'='*50}", flush=True)
        clf  = clfs[model_name]
        phi0 = float(clf.predict_proba(X_train)[:, 1].mean())
        pm   = make_predict_fn(clf)
        print(f"phi0 (mean train pred): {phi0:.4f}", flush=True)

        for s_idx, seed in enumerate(SEEDS):
            out_path = os.path.join(OUT_DIR,
                f'{DATASET}_{model_name}_ctree{FILE_SUFFIX}_s{s_idx}_sv.pkl')
            if os.path.exists(out_path):
                print(f"  [SKIP] s{s_idx} seed={seed}", flush=True)
                continue

            print(f"\n  seed={seed} (s{s_idx}/{len(SEEDS)-1})...", flush=True)
            t0 = time.time()
            result = pyshapr.explain(
                model         = clf,
                x_explain     = X_test,
                x_train       = X_train,
                approach      = APPROACH,
                phi0          = phi0,
                n_MC_samples  = N_MC,
                seed          = seed,
                predict_model = pm,
                verbose       = None,
            )
            elapsed = time.time() - t0
            sv = result.to_shap().values
            print(f"  Done {elapsed/60:.1f} min | shape {sv.shape}", flush=True)

            with open(out_path, 'wb') as f:
                pickle.dump({'values': sv, 'seed': seed,
                             'model': model_name, 'approach': APPROACH,
                             'n_MC_samples': N_MC, 'n_explain': len(X_test),
                             'feature_names': X_test.columns.tolist()}, f, protocol=4)
            print(f"  Saved -> {out_path}", flush=True)

    print("\nAll done.", flush=True)
