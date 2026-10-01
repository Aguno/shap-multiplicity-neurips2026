"""Appendix E consensus-rank and Top-1 summaries from saved K100 SHAP.

This module reads saved explanations only. It does not fit a model, predict,
generate SHAP values, recompute a baseline, or modify historical inputs.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import pickle
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "appendix_e"
DATASETS = ("german", "diabetes", "gmsc")
METHODS = ("random", "cte")
DATASET_LABELS = {"german": "German", "diabetes": "Diabetes", "gmsc": "GMSC"}
METHOD_LABELS = {"random": "Random", "cte": "CTE"}
EXPECTED_SHAPES = {"german": (198, 16), "diabetes": (79, 8), "gmsc": (9000, 10)}
PAIRS = tuple(itertools.combinations(range(5), 2))
# Same display aliases as the existing reference-query audit; order is preserved.
FEATURE_ALIASES = {
    "RevolvingUtilizationOfUnsecuredLines": "Revolving Util. (Unsecured)",
    "NumberOfTime30-59DaysPastDueNotWorse": "# of 30-59 DPD",
    "NumberOfOpenCreditLinesAndLoans": "# of Open Credit Lines",
    "NumberOfTimes90DaysLate": "# of 90 DPD",
    "NumberRealEstateLoansOrLines": "# of Real Estate Loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "# of 60-89 DPD",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_paths(dataset, method, seed):
    """Enumerate every required file; never glob a partial set or fall back."""
    stem = f"{dataset}_ftt_0_0_{method}_s{seed}"
    folder = ROOT / "results_tree"
    if dataset == "gmsc":
        return [folder / f"{stem}_c{chunk}_sv.pkl" for chunk in range(100)]
    return [folder / f"{stem}_sv.pkl"]


def expected_queries(dataset):
    """Reconstruct the historical fold indices, without fitting any encoder/model."""
    paths = [ROOT / f"dataset/{dataset}_{part}.pkl" for part in ("X", "Y")]
    with paths[0].open("rb") as handle:
        x = pickle.load(handle)
    with paths[1].open("rb") as handle:
        y = pickle.load(handle)
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    _, test_index = next(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(x, y))
    test = x.iloc[test_index]
    if test.shape != EXPECTED_SHAPES[dataset] or test.isna().any().any():
        raise ValueError(f"Unexpected shape or missing query values for {dataset}")
    evidence = {
        "dataset": dataset,
        "n_instances": len(test),
        "n_features": test.shape[1],
        "duplicate_feature_rows_beyond_first": int(test.duplicated().sum()),
        "query_alignment_scope": "Exact feature values and order; no claim of unique person IDs",
        "dataset_files": [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in paths],
    }
    return test, evidence


def load_canonical_runs():
    """Return six complete (5,N,d) signed arrays with source-file provenance."""
    runs, source_rows, run_rows, dataset_evidence = {}, [], [], []
    for dataset in DATASETS:
        expected, evidence = expected_queries(dataset)
        dataset_evidence.append(evidence)
        for method in METHODS:
            method_values = []
            for seed in range(5):
                paths = canonical_paths(dataset, method, seed)
                missing = [p for p in paths if not p.is_file()]
                if missing:
                    raise FileNotFoundError(f"Missing {len(missing)} files; first: {missing[0]}")
                arrays, queries = [], []
                for chunk_index, path in enumerate(paths):
                    with path.open("rb") as handle:
                        obj = pickle.load(handle)
                    values = np.asarray(obj.values)
                    query = np.asarray(obj.data)
                    names = getattr(obj, "feature_names", None)
                    if names is None:
                        raise ValueError(f"Missing feature names: {path}")
                    names = tuple(FEATURE_ALIASES.get(name, name) for name in names)
                    if names != tuple(expected.columns):
                        raise ValueError(f"Feature names/order mismatch: {path}")
                    if values.ndim != 2 or len(values) == 0 or query.shape != values.shape:
                        raise ValueError(f"Invalid attribution/query shape: {path}")
                    if not np.isfinite(values).all():
                        raise ValueError(f"Non-finite attributions: {path}")
                    arrays.append(values)
                    queries.append(query)
                    source_rows.append({
                        "dataset": dataset, "model": "ftt", "method": method,
                        "split": 0, "model_seed": 0, "background_size": 100,
                        "background_seed": seed,
                        "chunk": chunk_index if dataset == "gmsc" else "whole",
                        "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                        "n_instances": len(values), "n_features": values.shape[1],
                    })
                values = np.concatenate(arrays, axis=0)
                query = np.concatenate(queries, axis=0)
                if values.shape != EXPECTED_SHAPES[dataset]:
                    raise ValueError(f"Wrong total row/feature count for {dataset}/{method}/s{seed}")
                if query.shape != expected.shape or not np.array_equal(query, expected.to_numpy()):
                    raise ValueError(f"Query values/order mismatch for {dataset}/{method}/s{seed}")
                method_values.append(values)
                run_rows.append({
                    "dataset": dataset, "model": "ftt", "method": method,
                    "background_seed": seed, "n_files": len(paths),
                    "n_instances": len(values), "n_features": values.shape[1],
                    "query_values_and_order_verified": True, "feature_order_verified": True,
                    "finite_signed_attributions": True,
                })
            # Validation includes equal shapes, exactly five runs, and finite values.
            if len(method_values) != 5 or len({v.shape for v in method_values}) != 1:
                raise ValueError("Incomplete or differently shaped runs; truncation is forbidden")
            runs[(dataset, method)] = np.stack(method_values)
    return runs, pd.DataFrame(source_rows), pd.DataFrame(run_rows), dataset_evidence


def summarize(runs):
    """Reproduce J8 quicksort ranks and J4/J7 first-maximum Top-1 exactly."""
    rank_rows, top1_rows, checks, tie_rows = [], [], [], []
    old_path = ROOT / "figures_b/churn_analysis/summary.json"
    old = json.loads(old_path.read_text())
    for (dataset, method), signed in runs.items():
        n_runs, n_instances, n_features = signed.shape
        if n_runs != 5 or (n_instances, n_features) != EXPECTED_SHAPES[dataset]:
            raise ValueError("Unexpected run shape")
        absolute = np.abs(signed)
        mean_absolute = absolute.mean(axis=0)
        # np.argsort's historical default is quicksort. Retain it explicitly:
        # no stable/lexicographic/random tie-breaking substitution is made.
        consensus = np.argsort(-mean_absolute, axis=1, kind="quicksort")[:, :3]
        ranks = np.argsort(np.argsort(-absolute, axis=2, kind="quicksort"), axis=2, kind="quicksort")
        legacy_consensus = np.argsort(-mean_absolute, axis=1)[:, :3]
        legacy_ranks = np.argsort(np.argsort(-absolute, axis=2), axis=2)
        if not np.array_equal(consensus, legacy_consensus) or not np.array_equal(ranks, legacy_ranks):
            raise AssertionError("The J8 default-sort convention changed")
        for consensus_rank in range(3):
            positions = np.stack([
                ranks[seed, np.arange(n_instances), consensus[:, consensus_rank]] + 1
                for seed in range(5)
            ])
            counts = np.bincount(positions.reshape(-1), minlength=n_features + 1)[1:]
            denominator = 5 * n_instances
            if len(counts) != n_features or int(counts.sum()) != denominator:
                raise AssertionError("Consensus rank observations were lost")
            for actual_rank, count in enumerate(counts, 1):
                rank_rows.append({
                    "dataset": dataset, "model": "ftt", "method": method,
                    "background_size": 100, "consensus_rank": consensus_rank + 1,
                    "actual_rank": actual_rank, "count": int(count),
                    "denominator": denominator, "percent": 100.0 * count / denominator,
                    "n_instances": n_instances, "n_runs": 5,
                })
        # np.argmax picks the first index in a maximum tie, matching J4/J7.
        top1 = np.argmax(absolute, axis=2)
        any_count = int(np.count_nonzero(np.any(top1 != top1[[0]], axis=0)))
        pair_count = sum(int(np.count_nonzero(top1[a] != top1[b])) for a, b in PAIRS)
        row = {"dataset": dataset, "model": "ftt", "method": method, "background_size": 100,
               "n_instances": n_instances, "n_runs": 5, "n_pairs": 10,
               "anyseed_count": any_count, "anyseed_denominator": n_instances,
               "anyseed_percent": 100.0 * any_count / n_instances,
               "pairwise_count": pair_count, "pairwise_denominator": 10 * n_instances,
               "pairwise_percent": 100.0 * pair_count / (10 * n_instances)}
        top1_rows.append(row)
        for old_key, new_key in (("top1_anyseed", "anyseed_percent"), ("top1_pairwise", "pairwise_percent")):
            historical = old[old_key][dataset]["ftt"][method] * 100.0
            delta = abs(row[new_key] - historical)
            if delta > 1e-12:
                raise AssertionError(f"Historical summary mismatch: {dataset}/{method}/{old_key}")
            checks.append({"dataset": dataset, "method": method, "statistic": old_key,
                           "historical_percent": historical, "recomputed_percent": row[new_key],
                           "absolute_delta_percentage_points": delta})
        tie_rows.append({
            "dataset": dataset, "method": method,
            "run_query_top1_ties": int(np.count_nonzero((absolute == absolute.max(axis=2, keepdims=True)).sum(axis=2) > 1)),
            "consensus_query_top1_ties": int(np.count_nonzero((mean_absolute == mean_absolute.max(axis=1, keepdims=True)).sum(axis=1) > 1)),
            "run_query_any_magnitude_ties": int(np.count_nonzero(np.any(np.diff(np.sort(absolute, axis=2), axis=2) == 0, axis=2))),
        })
    return pd.DataFrame(rank_rows), pd.DataFrame(top1_rows), checks, tie_rows


def export_tables(rank_table, top1_table, source_table, run_table, dataset_evidence, checks, tie_rows):
    output = OUTPUT / "data"
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "churn_rank_frequencies.csv": rank_table,
        "churn_top1_summary.csv": top1_table,
        "churn_input_files.csv": source_table,
        "churn_run_validation.csv": run_table,
    }
    for name, table in tables.items():
        table.to_csv(output / name, index=False, float_format="%.15g")
    provenance = {
        "analysis": "FTT K100 consensus rank and Top-1 churn from existing signed SHAP",
        "explainer": "KernelSHAP", "split": 0, "model_seed": 0,
        "datasets": dataset_evidence, "methods": list(METHODS), "background_seeds": list(range(5)),
        "source_selection": "German/Diabetes whole; GMSC exact ordered c0 through c99; no fallback",
        "consensus": "Per-query mean absolute SHAP across five seeds; highest three features",
        "tie_convention": "J8 np.argsort quicksort for consensus and run ranks; J4/J7 np.argmax first maximum for Top1",
        "units": {"consensus_each_row": "5N run-query ranks", "anyseed": "N queries", "pairwise": "10N unordered-pair-query comparisons"},
        "known_german_random_query_187": "Included as in the existing authorized analysis; not excluded or reclassified",
        "inference_limit": "The same five runs select consensus features and supply their ranks; this is not independent-run generalization",
        "historical_sources": [
            {"path": "churn_analysis.ipynb", "cells_1_based": [4, 7, 8, 9, 10], "sha256": sha256(ROOT / "churn_analysis.ipynb")},
            {"path": "figures_b/churn_analysis/summary.json", "sha256": sha256(ROOT / "figures_b/churn_analysis/summary.json")},
        ],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "tie_diagnostics": tie_rows,
    }
    (output / "churn_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    validation = {
        "status": "passed", "source_file_count": len(source_table), "run_count": len(run_table),
        "configuration_count": len(top1_table), "rank_frequency_rows": len(rank_table),
        "complete_five_seeds": True, "complete_gmsc_chunks_0_to_99": True,
        "query_values_and_order_verified": bool(run_table.query_values_and_order_verified.all()),
        "feature_names_and_order_verified": bool(run_table.feature_order_verified.all()),
        "no_trimming_or_dropping": True, "all_18_consensus_rows_sum_to_5N": True,
        "all_18_consensus_percent_rows_sum_to_100": bool(np.allclose(rank_table.groupby(["dataset", "method", "consensus_rank"]).percent.sum(), 100, rtol=0, atol=1e-12)),
        "j8_explicit_quicksort_matches_original_expressions": True,
        "historical_top1_checks": checks,
        "source_hashes_verified_after_read": all(sha256(ROOT / row.path) == row.sha256 for row in source_table.itertuples()),
        "data_output_sha256": {name: sha256(output / name) for name in tables},
        "plot_color_scale": {"cmap": "YlOrRd", "minimum_percent": 0, "maximum_percent": 100},
    }
    if not validation["all_18_consensus_percent_rows_sum_to_100"] or not validation["source_hashes_verified_after_read"]:
        raise AssertionError("Post-export validation failed")
    (output / "churn_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    return validation


def make_figure(rank_table):
    """Plot every actual rank, including zero-frequency cells, on one scale."""
    import matplotlib.pyplot as plt
    from appendix_e.common import set_plot_style, save_figure

    set_plot_style()
    figure, axes = plt.subplots(3, 2, figsize=(8.8, 5.1), constrained_layout=True)
    for row_index, dataset in enumerate(DATASETS):
        for column_index, method in enumerate(METHODS):
            axis = axes[row_index, column_index]
            sub = rank_table[(rank_table.dataset == dataset) & (rank_table.method == method)]
            matrix = sub.pivot(index="consensus_rank", columns="actual_rank", values="percent").to_numpy()
            d = EXPECTED_SHAPES[dataset][1]
            image = axis.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100, interpolation="nearest")
            for i in range(3):
                for j in range(d):
                    percent = matrix[i, j]
                    # Whole percents keep the 16-column German panels legible.
                    # Positive frequencies below 0.5% are shown explicitly.
                    label = "0" if percent == 0 else ("<1" if percent < 0.5 else f"{percent:.0f}")
                    axis.text(j, i, label, ha="center", va="center", fontsize=13,
                              color="white" if percent > 55 else "black")
            axis.set_xticks(np.arange(d), [str(rank) for rank in range(1, d + 1)])
            axis.set_yticks([0, 1, 2], ["#1", "#2", "#3"])
            axis.tick_params(axis="both", which="both", length=0, labelsize=13)
            axis.set_xticks(np.arange(-0.5, d, 1), minor=True)
            axis.set_yticks(np.arange(-0.5, 3, 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=0.4)
            axis.set_title(f"{DATASET_LABELS[dataset]} · {METHOD_LABELS[method]}", pad=6, fontsize=16)
    figure.supylabel("Consensus rank", fontsize=15)
    figure.supxlabel("Rank in individual run", fontsize=15)
    colorbar = figure.colorbar(image, ax=axes, fraction=0.026, pad=0.022, ticks=[0, 20, 40, 60, 80, 100])
    colorbar.set_label("Run–query rank observations (%)", fontsize=15)
    colorbar.ax.tick_params(labelsize=13)
    save_figure(figure, "appendix_churn_depth")
    return figure
