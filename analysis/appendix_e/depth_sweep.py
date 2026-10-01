"""Redraw Appendix E depth and K sweeps from the saved B summary counts.

This module does not load SHAP arrays, train models, or simulate baselines.
The notebook entry points retain the original numerator, denominator and tau.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

from .common import (
    DATASET_LABELS, DATASET_ORDER, HEATMAP_CMAP, METHOD_COLORS, METHOD_LABELS,
    METHOD_MARKERS, METHOD_ORDER, MODEL_LABELS, save_figure, set_plot_style,
)

DATASETS = tuple(DATASET_ORDER)
MODELS = ("mlp", "ftt")
METHODS = tuple(METHOD_ORDER)
METRICS = ("jac", "l2", "rbo")
DEPTHS = (2, 3, 4, 5)
N_INSTANCES = {"german": 198, "diabetes": 79, "acs": 9192, "gmsc": 9000}
BACKGROUND_SIZES = {
    "german": (25, 50, 100, 200, 400, 790),
    "diabetes": (25, 50, 100, 200, 313),
    "acs": (25, 50, 100, 200, 400, 800, 1600),
    "gmsc": (25, 50, 100, 200, 400, 800, 1600),
}
FULL_TRAINING = {"german": 790, "diabetes": 313}
METRIC_LABELS = {"jac": "Top-3 Jaccard", "l2": r"$L_2$", "rbo": "1 − RBO"}
SOURCE_RELATIVE = {
    "topk": Path("figures_b/background_topk_table/pair_exceedance_counts.csv"),
    "sweep": Path("figures_b/background_sweep/pair_exceedance_counts.csv"),
    "main": Path("figures_b/background_table/pair_exceedance_counts.csv"),
}
COUNT_COLUMNS = ("n_exceed", "n_observations", "n_pairs", "n_instances")
VALUE_COLUMNS = (*COUNT_COLUMNS, "percent", "threshold")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_inputs(repo: Path, analysis: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the historical saved summary and the main table without float drift."""
    if analysis not in ("topk", "sweep"):
        raise ValueError("analysis must be topk or sweep")
    data = pd.read_csv(repo / SOURCE_RELATIVE[analysis], float_precision="round_trip")
    main = pd.read_csv(repo / SOURCE_RELATIVE["main"], float_precision="round_trip")
    main = main.loc[(main["explainer"] == "kernel_shap") & main["model"].isin(MODELS)].copy()
    return data, main


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_counts(data: pd.DataFrame, key_columns: list[str]) -> dict:
    _require(not data.empty, "Empty saved summary")
    _require(not data[key_columns].duplicated().any(), "Duplicate summary keys")
    _require(not data.isna().any().any(), "Missing summary value; no row is silently omitted")
    values = data[list(VALUE_COLUMNS)].to_numpy(dtype=float)
    _require(np.isfinite(values).all(), "Non-finite saved counts/rates/thresholds")
    for column in COUNT_COLUMNS:
        _require((data[column] == data[column].astype("int64")).all(), f"Non-integer {column}")
    _require((data["n_pairs"] == 10).all(), "B requires all ten unordered pairs")
    _require((data["n_instances"] > 0).all(), "Non-positive N")
    _require((data["n_instances"] == data["dataset"].map(N_INSTANCES)).all(), "Unexpected dataset N")
    _require((data["n_observations"] == 10 * data["n_instances"]).all(), "B denominator is not 10N")
    _require(data["n_exceed"].between(0, data["n_observations"]).all(), "Invalid numerator")
    recalculated = 100 * data["n_exceed"] / data["n_observations"]
    _require(np.array_equal(data["percent"].to_numpy(), recalculated.to_numpy()), "Saved percentage differs from 100*n_exceed/(10N)")
    _require((data["threshold"] > 0).all(), "Expected positive saved thresholds")
    if "status" in data:
        _require(data["status"].eq("complete").all(), "Incomplete source group")
    return {
        "rows": len(data), "unique_keys": True, "all_finite": True,
        "integer_counts": True, "n_pairs": 10, "denominator": "10N",
        "percentage_exact_from_counts": True, "max_percentage_reconstruction_error": 0.0,
        "counted_event": "distance > saved threshold (strict)",
        "validation_scope": "Saved-summary arithmetic, keys and cross-table equality; no raw distances or historical model identity are revalidated here.",
    }


def _compare_main(chosen: pd.DataFrame, main: pd.DataFrame, expected_rows: int) -> pd.DataFrame:
    keys = ["dataset", "model", "method", "metric"]
    expected_main = main.loc[main["metric"].isin(chosen["metric"].unique())]
    merged = chosen.merge(expected_main, on=keys, how="outer", suffixes=("_appendix", "_main"), validate="one_to_one", indicator=True)
    _require(len(merged) == expected_rows, "Unexpected main-table comparison row count")
    _require(merged["_merge"].eq("both").all(), "Main-table comparison has missing keys")
    result = merged[keys].copy()
    for column in VALUE_COLUMNS:
        same = merged[f"{column}_appendix"].eq(merged[f"{column}_main"])
        _require(same.all(), f"Main-table {column} differs")
        result[column] = merged[f"{column}_appendix"]
        result[f"{column}_exact_match"] = same
    return result


def validate_topk(data: pd.DataFrame, main: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    keys = ["dataset", "model", "method", "k"]
    report = _validate_counts(data, keys)
    expected = set(itertools.product(DATASETS, MODELS, METHODS, DEPTHS))
    _require(set(data[keys].itertuples(index=False, name=None)) == expected, "Top-k must contain all 128 requested combinations")
    _require(data.groupby(["dataset", "k"])["threshold"].nunique().eq(1).all(), "Threshold differs within a dataset/depth")
    _require(data["input_policy"].eq("main-table K100; no version fallback").all(), "Top-k source does not use canonical K100")
    expected_layout = data["dataset"].map({"german": "whole", "diabetes": "whole", "acs": "chunks_c0_c99", "gmsc": "chunks_c0_c99"})
    _require(data["input_layout"].eq(expected_layout).all(), "Unexpected canonical K100 input layout")
    chosen = data.loc[data["k"] == 3].assign(metric="jac")
    equivalence = _compare_main(chosen, main, 32)
    below_min = data["threshold"] < 2 / (data["k"] + 1)
    _require(below_min.all(), "A saved threshold is not below the minimum positive Jaccard distance")
    report.update({"complete_design_rows": 128, "depths": list(DEPTHS), "background_size": 100,
                   "fixed_threshold_within_dataset_depth": True, "main_k3_exact_match_rows": len(equivalence),
                   "all_saved_thresholds_below_minimum_positive_Jaccard": True,
                   "interpretation": "Only for these saved settings, strict baseline exceedance equals Top-k set disagreement. Different k use different baselines."})
    return report, equivalence


def validate_sweep(data: pd.DataFrame, main: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    keys = ["dataset", "model", "method", "bg_size", "metric"]
    report = _validate_counts(data, keys)
    expected = {(d, mo, me, k, mt) for d in DATASETS for mo in MODELS for me in METHODS for k in BACKGROUND_SIZES[d] for mt in METRICS}
    _require(set(data[keys].itertuples(index=False, name=None)) == expected, "Sweep must contain all 600 requested combinations")
    _require(data.groupby(["dataset", "metric"])["threshold"].nunique().eq(1).all(), "Threshold changes across K/model/method")
    equivalence = _compare_main(data.loc[data["bg_size"] == 100], main, 96)
    report.update({"complete_design_rows": 600, "background_sizes": {key: list(value) for key, value in BACKGROUND_SIZES.items()},
                   "fixed_threshold_across_K_within_dataset_metric": True, "main_K100_exact_match_rows": len(equivalence),
                   "full_training_K": FULL_TRAINING, "largest_requested_K": 1600,
                   "interpretation": "K=100 reuses the main canonical result. Segments join observed K values, without smoothing or confidence intervals."})
    return report, equivalence


def export_analysis(repo: Path, out: Path, analysis: str, data: pd.DataFrame, report: dict, equivalence: pd.DataFrame) -> dict:
    """Save portable plotting records and a provenance/validation sidecar."""
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    exported = data.copy()
    exported["dataset_label"] = exported["dataset"].map(DATASET_LABELS)
    exported["model_label"] = exported["model"].map(MODEL_LABELS)
    exported["method_label"] = exported["method"].map(METHOD_LABELS)
    exported["statistic"] = "100 * count(distance > threshold) / (10N)"
    exported["unit"] = "pair-instance comparisons (%)"
    exported["source_csv"] = str(SOURCE_RELATIVE[analysis])
    if analysis == "topk":
        exported["background_size"] = 100
        exported["metric"] = "jac"
        exported["minimum_positive_jaccard_distance"] = 2 / (exported["k"] + 1)
        compare_name = "topk_k3_main_equivalence.csv"
    else:
        exported["is_full_training"] = [FULL_TRAINING.get(d) == k for d, k in zip(exported["dataset"], exported["bg_size"])]
        exported["source_policy"] = np.where(exported["bg_size"].eq(100), "canonical main-table K100 reused", "saved background-size sweep")
        exported["requested_coalition_budget"] = exported["dataset"].map({"german": 1024, "diabetes": 256, "acs": 256, "gmsc": 1024})
        compare_name = "sweep_K100_main_equivalence.csv"
    csv_path = data_dir / f"{analysis}_plot_data.csv"
    exported.to_csv(csv_path, index=False)
    reread = pd.read_csv(csv_path, float_precision="round_trip")
    for column in data.columns:
        _require(reread[column].equals(data[column]), f"Portable CSV roundtrip changed {column}")
    equivalence.to_csv(data_dir / compare_name, index=False)
    provenance = {"analysis": analysis, "summary_source": str(SOURCE_RELATIVE[analysis]),
                  "sources": [{"path": str(SOURCE_RELATIVE[k]), "sha256": sha256(repo / SOURCE_RELATIVE[k])} for k in (analysis, "main")],
                  "plot_data": {"path": str(csv_path.relative_to(repo)), "sha256": sha256(csv_path), "rows": len(exported), "exact_original_column_roundtrip": True},
                  "validation": report, "new_experiments": False, "baseline_simulation": False,
                  "new_raw_SHAP_analysis": False, "original_summary_files_modified": False}
    (data_dir / f"{analysis}_validation.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def plot_topk(data: pd.DataFrame):
    """Four k panels, eight dataset/model rows, four methods, one shared scale."""
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(5.7, 5.5), layout="constrained")
    rows = [(d, mo) for d in DATASETS for mo in MODELS]
    short_dataset = {"german": "German", "diabetes": "Diabetes", "acs": "ACS", "gmsc": "GMSC"}
    row_names = [f"{short_dataset[d]} / {MODEL_LABELS[mo]}" for d, mo in rows]
    for panel_index, (ax, k) in enumerate(zip(axes.flat, DEPTHS)):
        selected = data.loc[data["k"] == k].set_index(["dataset", "model", "method"])
        matrix = np.array([[selected.loc[(d, mo, method), "percent"] for method in METHODS] for d, mo in rows])
        im = ax.imshow(matrix, cmap=HEATMAP_CMAP, vmin=0, vmax=100, aspect="auto", interpolation="none")
        ax.set_title(f"Top-{k}", fontsize=10, fontweight="bold", pad=6)
        ax.set_xticks(range(4), [METHOD_LABELS[method] for method in METHODS])
        ax.set_yticks(range(8), row_names if panel_index % 2 == 0 else [])
        ax.tick_params(axis="both", length=0, pad=4, labelsize=8.5)
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=8.5,
                    color="white" if value >= 60 else "#171717")
        ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)
        for cut in (1.5, 3.5, 5.5):
            ax.axhline(cut, color="#555555", linewidth=0.75)
        for spine in ax.spines.values():
            spine.set_visible(False)
    colorbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88, fraction=0.025, pad=0.025)
    colorbar.set_ticks([0, 25, 50, 75, 100])
    colorbar.ax.tick_params(labelsize=8.5)
    colorbar.set_label("Pair–instance comparisons exceeding the baseline (%)", labelpad=7, fontsize=9)
    return fig


def plot_background_size(data: pd.DataFrame, model: str, metric: str):
    """Observed-K line plots on a log2 axis; no extrapolation, smoothing or CI."""
    _require(model in MODELS and metric in METRICS, "Unknown requested model/metric")
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(5.9, 4.8), sharey=True)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.12, top=0.80, wspace=0.18, hspace=0.50)
    handles = []
    for ax, dataset in zip(axes.flat, DATASETS):
        subset = data.loc[(data["dataset"] == dataset) & (data["model"] == model) & (data["metric"] == metric)]
        for method in METHODS:
            rows = subset.loc[subset["method"] == method].sort_values("bg_size")
            line, = ax.plot(rows["bg_size"], rows["percent"], label=METHOD_LABELS[method],
                            color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                            markersize=4.5, linewidth=1.5, markeredgewidth=0.5,
                            markeredgecolor="white", clip_on=False)
            if dataset == DATASETS[0]:
                handles.append(line)
        actual_k = BACKGROUND_SIZES[dataset]
        ax.set_xscale("log", base=2)
        ax.set_xlim(actual_k[0] * 0.9, actual_k[-1] * 1.11)
        ax.set_xticks(actual_k, [f"{k}\n(full)" if FULL_TRAINING.get(dataset) == k else str(k) for k in actual_k])
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_title(DATASET_LABELS[dataset], fontsize=10, fontweight="bold", pad=7)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#dddddd", linewidth=0.65)
        ax.tick_params(axis="x", length=3, labelsize=9)
        ax.tick_params(axis="y", length=3, labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"{MODEL_LABELS[model]} · {METRIC_LABELS[metric]}", fontsize=11, fontweight="bold", y=0.99)
    fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.55, 0.95), frameon=False, columnspacing=1.1, handlelength=1.8, fontsize=9)
    fig.supxlabel("Background size K", fontsize=10, y=0.025)
    fig.supylabel("Pair–instance comparisons exceeding the baseline (%)", fontsize=10, x=0.012)
    return fig
