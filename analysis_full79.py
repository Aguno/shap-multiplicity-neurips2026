"""
analysis_full79.py
==================
Full-79 Conditional (ctree) vs Same-model Marginal (KernelSHAP/random) churn comparison.
Both estimators explain identical CPU-trained models (same monkey-patch + grid search).
Diabetes, split=0, models {FTT, MLP}, seeds {0,42,84,126,168}.

TABLE 1 only — no representativeness table (full fold = no subsample to compare).
Model label: 'identical CPU-trained models explained by both estimators'
"""

import os, sys, pickle, itertools, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def compute_jaccard_distance(v1, v2, k=3):
    """Jaccard distance (1 - Jaccard similarity) of top-k feature sets. Returns (N,) array."""
    top_k_v1 = np.argsort(np.abs(v1), axis=1)[:, -k:]
    top_k_v2 = np.argsort(np.abs(v2), axis=1)[:, -k:]
    n = v1.shape[0]
    dists = np.zeros(n)
    for i in range(n):
        s1 = set(top_k_v1[i]); s2 = set(top_k_v2[i])
        inter = len(s1 & s2); union = len(s1 | s2)
        dists[i] = 1.0 - inter / union if union > 0 else 0.0
    return dists

MODELS   = ['ftt', 'mlp']
SEEDS    = [0, 42, 84, 126, 168]
PAIRS    = list(itertools.combinations(range(5), 2))
K        = 3
JAC_VALS = [0.0, 0.5, 0.8, 1.0]

COND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_conditional')   # conditional: *_ctree_full79_s{i}_sv.pkl
MARG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_conditional')   # marginal:    *_random_cpumodel_s{i}_sv.pkl
N_FULL   = 79

# ── 1. Load conditional SHAP values (full-79) ─────────────────────────────────
print("Loading conditional (ctree, full-79) SHAP values...")
cond_sv = {}
for model in MODELS:
    cond_sv[model] = {}
    for s_idx in range(5):
        path = os.path.join(COND_DIR, f'diabetes_{model}_ctree_full79_s{s_idx}_sv.pkl')
        if not os.path.exists(path):
            print(f"  [MISSING] {path}")
            continue
        with open(path, 'rb') as f:
            d = pickle.load(f)
        sv = d['values']
        assert sv.shape == (N_FULL, 8), f"{path}: shape {sv.shape}"
        cond_sv[model][s_idx] = sv
print(f"  OK: {sum(len(v) for v in cond_sv.values())} / {len(MODELS)*5} seeds loaded")

# ── 2. Load marginal SHAP values (same-model CPU) ─────────────────────────────
print("\nLoading marginal (KernelSHAP/random, same CPU model) SHAP values...")
marg_sv = {}
for model in MODELS:
    marg_sv[model] = {}
    for s_idx in range(5):
        path = os.path.join(MARG_DIR, f'diabetes_{model}_0_0_random_cpumodel_s{s_idx}_sv.pkl')
        if not os.path.exists(path):
            print(f"  [MISSING] {path}")
            continue
        with open(path, 'rb') as f:
            sv_obj = pickle.load(f)
        sv = sv_obj.values   # (79, 8)
        assert sv.shape == (N_FULL, 8), f"{path}: shape {sv.shape}"
        marg_sv[model][s_idx] = sv
print(f"  OK: {sum(len(v) for v in marg_sv.values())} / {len(MODELS)*5} seeds loaded")

# ── 3. Metric helpers ──────────────────────────────────────────────────────────
def round_jac(v):
    return min(JAC_VALS, key=lambda x: abs(x - v))

def top1_churn(sv_dict):
    """Given sv_dict[s] = (N,8) array, return (any_pct, pairwise_pct)."""
    top1s = [np.argmax(np.abs(sv_dict[s]), axis=1) for s in range(5)]
    top1_mat = np.stack(top1s, axis=0)   # (5, N)
    any_disagree = ~np.all(top1_mat == top1_mat[[0], :], axis=0)
    pair_arr = np.concatenate([(top1s[i] != top1s[j]).astype(float) for i,j in PAIRS])
    return float(any_disagree.mean()), float(pair_arr.mean())

def jac_props(sv_dict):
    """Given sv_dict[s] = (N,8) array, return {jac_val: proportion}."""
    all_jac = np.concatenate([
        compute_jaccard_distance(sv_dict[i], sv_dict[j], k=K)
        for i, j in PAIRS
    ])
    counts = {v: 0 for v in JAC_VALS}
    for val in all_jac:
        counts[round_jac(float(val))] += 1
    total = len(all_jac)
    return {v: counts[v]/total for v in JAC_VALS}

# ── 4. Compute metrics ─────────────────────────────────────────────────────────
print("\nComputing metrics...")
rows = []
for model in MODELS:
    for method_name, sv_dict in [
            ('ctree (conditional)',    cond_sv[model]),
            ('random (marginal)',      marg_sv[model]),
    ]:
        if len(sv_dict) < 5:
            print(f"  [SKIP] {model}/{method_name}: only {len(sv_dict)}/5 seeds")
            continue
        any_c, pair_c = top1_churn(sv_dict)
        jp = jac_props(sv_dict)
        rows.append({
            'method': method_name,
            'model': model.upper(),
            'any_seed': any_c,
            'pairwise': pair_c,
            'jac_0':   jp[0.0],
            'jac_05':  jp[0.5],
            'jac_08':  jp[0.8],
            'jac_10':  jp[1.0],
        })

# ── 5. Print TABLE 1 ──────────────────────────────────────────────────────────
def pct(v):  return f"{v*100:.1f}%"

print("\n" + "="*90)
print("TABLE 1 — Conditional (ctree) vs Marginal (KernelSHAP/random) Churn Comparison")
print(f"Dataset: Diabetes, N=79 (full test fold), K={K}")
print("Model: identical CPU-trained models explained by both estimators")
print("="*90)
print(f"{'Method':<28} {'Model':<6} {'any-seed':<10} {'pairwise':<10} {'Jac=0':>8} {'Jac=0.5':>8} {'Jac=0.8':>8} {'Jac=1.0':>8}")
print("-"*90)
for r in rows:
    print(f"{r['method']:<28} {r['model']:<6} {pct(r['any_seed']):<10} {pct(r['pairwise']):<10} "
          f"{pct(r['jac_0']):>8} {pct(r['jac_05']):>8} {pct(r['jac_08']):>8} {pct(r['jac_10']):>8}")

print(f"\nNote: Jac=0 = identical top-{K} sets. Jac=1.0 = completely disjoint (worst).")

# ── 6. Summary paragraph ──────────────────────────────────────────────────────
def mavg_method(rows_list, method_substr):
    sel = [r for r in rows_list if method_substr in r['method']]
    if not sel: return None
    return {k: np.mean([r[k] for r in sel])
            for k in ['any_seed','pairwise','jac_0','jac_05','jac_08','jac_10']}

cond_avg = mavg_method(rows, 'ctree')
marg_avg = mavg_method(rows, 'marginal')

if cond_avg and marg_avg:
    delta_any  = cond_avg['any_seed'] - marg_avg['any_seed']
    delta_pair = cond_avg['pairwise'] - marg_avg['pairwise']
    delta_jac0 = cond_avg['jac_0']    - marg_avg['jac_0']
    direction = ("reduces" if delta_any < -0.02
                 else ("increases" if delta_any > 0.02 else "is comparable to"))

    print("\n" + "="*90)
    print("SUMMARY PARAGRAPH")
    print("="*90)
    print(
        f"On the full Diabetes test fold (N=79, fold-0), identical CPU-trained models\n"
        f"(same grid-selected hyperparameters) explained by conditional SHAP (ctree, n_MC=1000,\n"
        f"5 background seeds) vs marginal KernelSHAP (random background, K=100, nsamples=256):\n\n"
        f"Model-averaged any-seed top-1 churn: ctree={pct(cond_avg['any_seed'])}, "
        f"marginal={pct(marg_avg['any_seed'])} (Δ={delta_any*100:+.1f} pp) — "
        f"conditional {direction} marginal churn.\n"
        f"Pairwise top-1 churn: ctree={pct(cond_avg['pairwise'])}, "
        f"marginal={pct(marg_avg['pairwise'])} (Δ={delta_pair*100:+.1f} pp).\n"
        f"Jac=0 (identical top-{K} sets): ctree={pct(cond_avg['jac_0'])}, "
        f"marginal={pct(marg_avg['jac_0'])} (Δ={delta_jac0*100:+.1f} pp)."
    )
