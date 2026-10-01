"""
experiment_marginal_cpu.py
==========================
Marginal KernelSHAP on checkpoint-loaded FTT/MLP for Diabetes.
Loads models_cpu/diabetes_{model}_seed0.pt — no training, no grid search.

Output: results_conditional/diabetes_{model}_0_0_random_cpumodel_s{i}_sv.pkl
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
import shap
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_data

DATASET        = 'diabetes'
MODELS_RUN     = ['ftt', 'mlp']
BG_SEEDS       = [0, 42, 84, 126, 168]
BG_SIZE        = 100
NSAMPLES       = 256
EXPLAINER_SEED = 42
SPLIT_SEED     = 0
OUT_DIR        = 'results_conditional'
CKPT_DIR       = 'models_cpu'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load & split ──────────────────────────────────────────────────────────────
X, Y = load_data(DATASET)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for i, (tr, te) in enumerate(skf.split(X, Y)):
    if i == SPLIT_SEED: break

X_train = X.iloc[tr].reset_index(drop=True)
X_test  = X.iloc[te].reset_index(drop=True)
raw_cols = X.columns.tolist()

print(f"Diabetes split=0: train={len(X_train)}, test={len(X_test)}")
print(f"torch threads: {torch.get_num_threads()}", flush=True)

# ── Checkpoint loader ─────────────────────────────────────────────────────────
def load_checkpoint(model_name):
    path = os.path.join(CKPT_DIR, f'{DATASET}_{model_name}_seed0.pt')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    return ckpt['clf'], ckpt['verify_pred3']

# ── Predict function (ndarray → proba) ───────────────────────────────────────
def make_predict_fn(clf):
    def f(X_arr):
        X_df = pd.DataFrame(X_arr, columns=raw_cols) if isinstance(X_arr, np.ndarray) else X_arr
        return clf.predict_proba(X_df)[:, 1]
    return f

def bg_random(seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_train), BG_SIZE, replace=False)
    return X_train.iloc[idx]

def result_path(model_name, s_idx):
    return os.path.join(OUT_DIR,
        f'{DATASET}_{model_name}_0_0_random_cpumodel_s{s_idx}_sv.pkl')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['ftt', 'mlp', 'both'], default='both')
    args = parser.parse_args()
    models_to_run = MODELS_RUN if args.model == 'both' else [args.model]

    # ── Load & verify checkpoints ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("CHECKPOINT VERIFICATION")
    print("="*60)
    clfs = {}
    for model_name in models_to_run:
        clf, saved_pred3 = load_checkpoint(model_name)
        live_pred3 = clf.predict_proba(X_test.iloc[:3])[:, 1]
        match = np.array_equal(live_pred3, saved_pred3)
        print(f"  [{model_name.upper()}] test[:3]: {np.round(live_pred3, 8).tolist()}")
        print(f"         vs saved:    {np.round(saved_pred3, 8).tolist()}")
        print(f"         byte-match:  {'PASS ✓' if match else 'FAIL ✗'}", flush=True)
        clfs[model_name] = clf

    # ── Timing probe: 1 instance KernelSHAP ──────────────────────────────────
    print("\n" + "="*60)
    print("TIMING: 1 instance KernelSHAP (no training)")
    print("="*60)
    probe_model = 'ftt' if 'ftt' in clfs else models_to_run[0]
    f_probe = make_predict_fn(clfs[probe_model])
    bg_probe = bg_random(BG_SEEDS[0])

    t0 = time.time()
    exp_probe = shap.KernelExplainer(f_probe, bg_probe, link='identity', seed=EXPLAINER_SEED)
    sv_probe  = exp_probe.shap_values(X_test.values[:1], nsamples=NSAMPLES, gc_collect=True)
    shap_per  = time.time() - t0
    print(f"  KernelSHAP per instance: {shap_per:.2f}s", flush=True)

    n_models  = len(models_to_run)
    total_est = shap_per * len(X_test) * len(BG_SEEDS) * n_models
    print(f"  Extrapolated ({len(X_test)} inst × {len(BG_SEEDS)} seeds × {n_models} models): "
          f"{total_est/3600:.2f}h", flush=True)
    del exp_probe, sv_probe

    # ── Full experiment ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FULL MARGINAL (checkpoint model)")
    print("="*60)

    for model_name in models_to_run:
        print(f"\n{'='*50}\nModel: {model_name.upper()}\n{'='*50}", flush=True)

        if all(os.path.exists(result_path(model_name, s)) for s in range(len(BG_SEEDS))):
            print(f"  [SKIP] all seeds exist")
            continue

        clf = clfs[model_name]
        f   = make_predict_fn(clf)

        for s_idx, bg_seed in enumerate(BG_SEEDS):
            out_path = result_path(model_name, s_idx)
            if os.path.exists(out_path):
                print(f"  [SKIP] s{s_idx} bg_seed={bg_seed}", flush=True)
                continue

            print(f"\n  bg_seed={bg_seed} (s{s_idx}/{len(BG_SEEDS)-1})...", flush=True)
            bg_df = bg_random(bg_seed)

            t0 = time.time()
            explainer = shap.KernelExplainer(f, bg_df, link='identity', seed=EXPLAINER_SEED)
            sv_raw    = explainer.shap_values(X_test.values, nsamples=NSAMPLES, gc_collect=True)
            elapsed   = time.time() - t0

            sv = sv_raw[1] if isinstance(sv_raw, list) else sv_raw
            bv = explainer.expected_value
            base_val = float(bv[1] if hasattr(bv, '__len__') else bv)

            err = np.abs(sv.sum(axis=1) + base_val - f(X_test.values)).mean()
            print(f"  Done {elapsed/60:.1f} min | shape {sv.shape} | err {err:.6f}", flush=True)

            shap_exp = shap.Explanation(values=sv, base_values=base_val,
                                        data=X_test.values, feature_names=raw_cols)
            with open(out_path, 'wb') as fout:
                pickle.dump(shap_exp, fout, protocol=4)
            print(f"  Saved -> {out_path}", flush=True)
            del explainer, sv_raw, sv, shap_exp

    print("\nAll done.", flush=True)
