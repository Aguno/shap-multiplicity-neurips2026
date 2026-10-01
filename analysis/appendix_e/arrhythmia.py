"""Reproduce the approved Arrhythmia raw-churn appendix from saved SHAP only.

No model fitting, SHAP generation, Monte Carlo, or calibration is performed.
Historical ranking conventions are retained and tested for boundary ties.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ("mlp", "ftt")
METHODS = ("random", "cte")
SEEDS = tuple(range(5))
PAIRS = tuple(itertools.combinations(SEEDS, 2))
N, D, EPS = 84, 278, 1e-10
DENOMINATOR = len(PAIRS) * N


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_sha256(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def find_project_root(start=None):
    """Locate the checkout from this script or an ancestor of the working dir."""
    starts = [Path(start or Path.cwd()).resolve(), Path(__file__).resolve().parent]
    for candidate in starts:
        for root in (candidate, *candidate.parents):
            if (root / ".shap_multiplicity_root").is_file() and (root / "results_tree").is_dir():
                return root
    raise FileNotFoundError("Run inside the framework checkout containing results_tree.")


def load_saved_runs(root):
    """Require all twenty files, fixed shapes, finite data and exact alignment."""
    import pickle

    root = Path(root)
    runs, sources, checks = {}, [], []
    reference_data = reference_names = None
    for model, method in itertools.product(MODELS, METHODS):
        arrays = []
        for seed in SEEDS:
            path = root / "results_tree" / f"arrhythmia_{model}_0_0_{method}_s{seed}_sv.pkl"
            before = sha256(path)  # Missing files fail here; there is no fallback or skip.
            with path.open("rb") as handle:
                obj = pickle.load(handle)
            get = obj.get if isinstance(obj, dict) else lambda key: getattr(obj, key, None)
            values, data = np.asarray(get("values")), np.asarray(get("data"))
            names = list(get("feature_names"))
            assert values.shape == data.shape == (N, D), (path, values.shape, data.shape)
            assert np.isfinite(values).all() and np.isfinite(data).all(), path
            assert len(names) == D and len(set(names)) == D, path
            assert np.isfinite(np.asarray(get("base_values"))).all(), path
            if reference_data is None:
                reference_data, reference_names = data.copy(), names
            assert names == reference_names, f"Feature order differs: {path}"
            assert np.array_equal(data, reference_data), f"Query values/order differ: {path}"
            absolute = np.abs(values)
            exact_support = np.count_nonzero(values, axis=1)
            eps_support = np.count_nonzero(absolute > EPS, axis=1)
            assert np.all(exact_support == 10) and np.all(eps_support == 10), path
            ordered = np.sort(absolute, axis=1)
            top1_ties = int(np.sum((absolute == absolute.max(axis=1)[:, None]).sum(axis=1) > 1))
            top3_boundary_ties = int(np.sum(ordered[:, -3] == ordered[:, -4]))
            ascending = np.sort(np.argsort(absolute, axis=1)[:, -3:], axis=1)
            descending = np.sort(np.argsort(-absolute, axis=1)[:, :3], axis=1)
            top3_conventions_differ = int(np.any(ascending != descending, axis=1).sum())
            # All observed top decisions are tie-free; do not silently adopt a new tie rule.
            assert top1_ties == top3_boundary_ties == top3_conventions_differ == 0, path
            assert sha256(path) == before, f"Source changed during read: {path}"
            sources.append({
                "path": str(path.relative_to(root)), "sha256": before,
                "model": model, "method": method, "seed": seed,
                "shape": [N, D], "data_sha256": _array_sha256(data),
                "feature_names_sha256": hashlib.sha256(json.dumps(names).encode()).hexdigest(),
            })
            checks.append({
                "model": model, "method": method, "seed": seed,
                "queries": N, "features": D, "nonzero_min": 10, "nonzero_max": 10,
                "top1_ties": top1_ties, "top3_boundary_ties": top3_boundary_ties,
                "top3_convention_set_differences": top3_conventions_differ,
                "query_and_feature_order_aligned": True,
            })
            arrays.append(values)
        runs[(model, method)] = np.stack(arrays)
    assert len(sources) == 20 and len(checks) == 20
    return runs, sources, pd.DataFrame(checks)


def _historical_jaccard(root):
    """Load only the pure historical distance function; never execute the module."""
    path = Path(root) / "plot_violin.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "compute_jaccard_distance")
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"np": np}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["compute_jaccard_distance"]


def compute_metrics(root, runs):
    """Return exact integer event counts, raw histogram counts, and support means."""
    historical_jaccard = _historical_jaccard(root)
    counts, observations, histograms, support_histograms = [], [], [], []
    historical_checks = []
    for model, method in itertools.product(MODELS, METHODS):
        values = runs[(model, method)]
        assert values.shape == (5, N, D)
        absolute = np.abs(values)
        top1 = np.argmax(absolute, axis=2)  # Historical any-seed/pairwise convention.
        top3 = np.argsort(absolute, axis=2)[:, :, -3:]  # Historical raw Jaccard convention.
        top3_desc = np.argsort(-absolute, axis=2)[:, :, :3]  # Historical decomposition convention.
        support = absolute > EPS
        n_any = int(np.count_nonzero(~np.all(top1 == top1[[0], :], axis=0)))
        combo_rows = []
        for si, sj in PAIRS:
            historical = historical_jaccard(values[si], values[sj], k=3)
            for query in range(N):
                left, right = set(top3[si, query]), set(top3[sj, query])
                inter = len(left & right)
                distance = 1 - inter / (6 - inter)
                assert distance == historical[query]
                left_desc, right_desc = set(top3_desc[si, query]), set(top3_desc[sj, query])
                assert left == left_desc and right == right_desc
                differing = left_desc ^ right_desc
                selection = any(not (support[si, query, f] and support[sj, query, f])
                                for f in differing)
                rank = any(support[si, query, f] and support[sj, query, f] for f in differing)
                support_intersection = int(np.count_nonzero(support[si, query] & support[sj, query]))
                support_union = int(np.count_nonzero(support[si, query] | support[sj, query]))
                assert support_union == 20 - support_intersection and support_union > 0
                row = {
                    "model": model, "method": method, "seed_i": si, "seed_j": sj,
                    "query_position": query, "top1_disagrees": int(top1[si, query] != top1[sj, query]),
                    "top3_intersection": inter, "top3_jaccard_distance": distance,
                    "top3_disagrees": int(inter < 3), "selection_churn": int(selection),
                    "rank_churn": int(rank), "selection_and_rank": int(selection and rank),
                    "support_intersection": support_intersection, "support_union": support_union,
                    "support_jaccard_similarity": support_intersection / support_union,
                }
                assert (selection or rank) == (inter < 3)
                combo_rows.append(row)
        frame = pd.DataFrame(combo_rows)
        assert len(frame) == DENOMINATOR == 840
        n_top3 = int(frame.top3_disagrees.sum())
        n_sel, n_rank = int(frame.selection_churn.sum()), int(frame.rank_churn.sum())
        n_both = int(frame.selection_and_rank.sum())
        assert n_sel + n_rank - n_both == n_top3
        row = {
            "model": model, "method": method, "N": N, "seeds": 5, "pairs": 10,
            "pair_denominator": DENOMINATOR,
            "top1_pair_numerator": int(frame.top1_disagrees.sum()),
            "top1_pair_fraction": float(frame.top1_disagrees.mean()),
            "top1_pair_percent": 100 * float(frame.top1_disagrees.mean()),
            "top3_disagreement_numerator": n_top3,
            "top3_disagreement_fraction": n_top3 / DENOMINATOR,
            "top3_disagreement_percent": 100 * n_top3 / DENOMINATOR,
            "top1_any_numerator": n_any, "top1_any_denominator": N,
            "top1_any_fraction": n_any / N,
            "selection_numerator": n_sel, "rank_numerator": n_rank,
            "selection_and_rank_numerator": n_both, "selection_rank_denominator": n_top3,
            "selection_churn_fraction": n_sel / n_top3, "rank_churn_fraction": n_rank / n_top3,
            "support_mean_jaccard_similarity": float(frame.support_jaccard_similarity.mean()),
            "support_mean_jaccard_distance": 1 - float(frame.support_jaccard_similarity.mean()),
            "support_denominator": DENOMINATOR,
        }
        counts.append(row)
        observations.extend(combo_rows)
        for intersection in (3, 2, 1, 0):
            numerator = int((frame.top3_intersection == intersection).sum())
            histograms.append({
                "model": model, "method": method, "intersection_size": intersection,
                "replaced_features": 3 - intersection,
                "jaccard_distance": 1 - intersection / (6 - intersection),
                "numerator": numerator, "denominator": DENOMINATOR,
                "fraction": numerator / DENOMINATOR, "percent": 100 * numerator / DENOMINATOR,
            })
        for intersection in range(11):
            numerator = int((frame.support_intersection == intersection).sum())
            support_histograms.append({
                "model": model, "method": method, "support_intersection": intersection,
                "support_union": 20 - intersection, "numerator": numerator,
                "denominator": DENOMINATOR, "similarity": intersection / (20 - intersection),
            })
        historical_checks.append({"model": model, "method": method,
                                  "jaccard_exactly_matches_historical_function": True,
                                  "comparisons": DENOMINATOR})
    return {
        "counts": pd.DataFrame(counts), "pair_observations": pd.DataFrame(observations),
        "jaccard_histogram": pd.DataFrame(histograms),
        "support_histogram": pd.DataFrame(support_histograms),
        "historical_function_checks": historical_checks,
    }


def validate_existing_summary(root, counts):
    """Reproduce all twenty stored any-seed/pairwise/support/conditional values."""
    old = json.loads((Path(root) / "figures_b/arrhythmia_analysis/summary.json").read_text())
    mapping = {
        "top1_any": "top1_any_fraction", "top1_pair": "top1_pair_fraction",
        "selection_churn": "selection_churn_fraction", "rank_churn": "rank_churn_fraction",
        "support_jaccard": "support_mean_jaccard_similarity",
    }
    checks = []
    for row in counts.to_dict("records"):
        for previous, current in mapping.items():
            expected, actual = old[row["model"]][row["method"]][previous], row[current]
            delta = abs(actual - expected)
            assert delta < 5e-15, (row["model"], row["method"], previous, actual, expected)
            checks.append({"model": row["model"], "method": row["method"], "metric": previous,
                           "previous": expected, "recomputed": actual, "absolute_difference": delta})
    assert len(checks) == 20
    return pd.DataFrame(checks)


def export_results(root, results, sources, checks, historical_comparison):
    root = Path(root)
    output = root / "appendix_e/data"
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "event_counts": results["counts"], "pair_observations": results["pair_observations"],
        "jaccard_histogram": results["jaccard_histogram"], "support_histogram": results["support_histogram"],
        "input_checks": checks, "existing_summary_comparison": historical_comparison,
    }
    for name, table in tables.items():
        table.to_csv(output / f"arrhythmia_{name}.csv", index=False, float_format="%.17g")
    dependencies = ["arrhythmia_analysis.ipynb", "plot_violin.py",
                    "figures_b/arrhythmia_analysis/summary.json", "analysis/appendix_e/arrhythmia.py",
                    "analysis/appendix_e/common.py"]
    provenance = {
        "source_files": sources,
        "analysis_sources": [{"path": p, "sha256": sha256(root / p)} for p in dependencies],
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "configuration": {"dataset": "arrhythmia", "N": N, "features": D, "K": 100,
                          "seeds": list(SEEDS), "pairs": [list(p) for p in PAIRS],
                          "support_epsilon": EPS, "pair_denominator": DENOMINATOR},
        "alignment": "Exact raw query values/order and feature names/order across all 20 files.",
        "tie_conventions": {
            "top1": "numpy.argmax(abs(values)); first maximum in the feature order",
            "top3_jaccard": "numpy.argsort(abs(values), axis=1)[:, -3:] with NumPy default sorting",
            "selection_rank": "numpy.argsort(-abs(values), axis=1)[:, :3] with NumPy default sorting",
            "observed_boundary_ties": 0,
        },
        "scope": "Saved-SHAP analysis only. No nominal-d or effective-d Mallows calibration; no training, SHAP generation or Monte Carlo.",
        "interpretation": "Top-1 and Top-3 figure panels are raw event rates, not calibrated threshold exceedance rates. Conditional selection/rank categories overlap.",
    }
    (output / "arrhythmia_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "arrhythmia_summary.json").write_text(json.dumps({
        "rows": results["counts"].to_dict("records"),
        "top3_jaccard_histogram": results["jaccard_histogram"].to_dict("records"),
        "unit": "run-pair--instance comparisons", "pair_denominator": DENOMINATOR,
    }, indent=2) + "\n")
    validation = {
        "status": "passed", "source_files": len(sources), "groups": 4,
        "seed_query_explanations": 4 * 5 * N, "pair_query_observations": 4 * DENOMINATOR,
        "all_shapes": [5, N, D], "raw_query_and_feature_alignment": True,
        "all_support_sizes_exact_and_eps": 10, "top1_and_top3_boundary_ties": 0,
        "historical_top3_function": results["historical_function_checks"],
        "historical_summary_values_matched": len(historical_comparison),
        "historical_summary_max_abs_difference": float(historical_comparison.absolute_difference.max()),
        "histograms_sum_to_840": bool((results["jaccard_histogram"].groupby(["model", "method"]).numerator.sum() == DENOMINATOR).all()),
        "source_hashes_still_match": all(sha256(root / s["path"]) == s["sha256"] for s in sources),
        "calibration_applied": False, "training_or_shap_generation_performed": False,
    }
    assert validation["histograms_sum_to_840"] and validation["source_hashes_still_match"]
    (output / "arrhythmia_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    return output, validation


def plot_churn(root, counts):
    """Two shared-scale heatmaps of raw events; no support or calibrated panel."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from appendix_e.common import HEATMAP_CMAP, set_plot_style, save_figure

    set_plot_style()
    with plt.rc_context({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 9, "ytick.labelsize": 9}):
        fig = plt.figure(figsize=(5.5, 2.2), layout="constrained")
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.065], wspace=0.15)
        axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
        cax = fig.add_subplot(gs[0, 2])
        normalization = Normalize(0, 100)
        cmap = plt.get_cmap(HEATMAP_CMAP)
        for ax, column, title in zip(axes, ["top1_pair_percent", "top3_disagreement_percent"],
                                     ["(a) Top-1 pairwise churn", "(b) Top-3 set disagreement"]):
            matrix = counts.pivot(index="model", columns="method", values=column).loc[list(MODELS), list(METHODS)].to_numpy()
            image = ax.imshow(matrix, cmap=cmap, norm=normalization, aspect="auto")
            ax.set_xticks([0, 1], ["Random", "CTE"])
            ax.set_yticks([0, 1], ["MLP", "FTT"])
            ax.set_title(title, pad=9)
            ax.tick_params(length=0, pad=5)
            ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.5)
            ax.tick_params(which="minor", bottom=False, left=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            for row in range(2):
                for col in range(2):
                    value = matrix[row, col]
                    ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=10,
                            color="white" if value >= 60 else "black")
        colorbar = fig.colorbar(image, cax=cax, ticks=[0, 20, 40, 60, 80, 100])
        colorbar.set_label("Pair–instance comparisons (%)", fontsize=8, labelpad=6)
        colorbar.ax.tick_params(labelsize=8)
        save_figure(fig, "appendix_arrhythmia_churn")
    return fig


def run(root=None):
    root = find_project_root(root)
    runs, sources, checks = load_saved_runs(root)
    results = compute_metrics(root, runs)
    comparison = validate_existing_summary(root, results["counts"])
    output, validation = export_results(root, results, sources, checks, comparison)
    figure = plot_churn(root, results["counts"])
    print(results["counts"][["model", "method", "top1_pair_numerator", "top3_disagreement_numerator", "pair_denominator", "top1_pair_percent", "top3_disagreement_percent"]].to_string(index=False))
    print(json.dumps(validation, indent=2))
    return figure


if __name__ == "__main__":
    import sys
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    run()
