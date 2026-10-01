# Explanation Multiplicity in SHAP: Characterization and Assessment

Code and analysis accompanying the NeurIPS 2026 paper by Hyunseung Hwang, Seungeun Lee, Lucas Rosenblatt, Steven Euijong Whang, and Julia Stoyanovich.

This package contains the paper's current pairwise analysis, figure notebooks, and preserved experiment generators. The OSF project for precomputed explanations, predictions, datasets, and checkpoints is [here](https://osf.io/qa9fk/); see [data setup](docs/DATA.md) for the archive layout and extraction instructions.

## Main statistic

With five explanation runs for each of N test queries, the main reported percentage is

```text
100 * count(distance[run_pair, query] > threshold) / (10 * N)
```

The ten unordered run pairs are evaluated separately. Distances are not averaged within a query before thresholding. The strict comparison is `>`, not `>=`. Full-background comparisons instead have five distances per query and use `5N`. Any-seed Top-1 churn uses `N`; pairwise Top-1 churn uses `10N`; consensus-rank summaries use `5N` for each selected consensus rank. These diagnostics retain their own definitions.

The current RBO implementation includes the extrapolation term and uses the full feature depth with `p = 1 - 1/d`. For K100, the active loaders use whole German/Diabetes outputs and complete ACS/GMSC chunks. The background sweep stops at K1600, or the available training size for smaller datasets.

## Analyze precomputed results

Use Python 3.12 and a separate environment. The analysis requirement pins are drawn from the preserved research environment exports; packaging has not installed a new environment or rerun every notebook.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
jupyter lab
```

Extract the required archives into this directory first. Keep the directory names inside each ZIP; in particular, the stored prediction path is `proba/proba/`. The `.shap_multiplicity_root` marker identifies the project root for the portable analysis notebooks.

| Notebook or script | Purpose | Required input folders |
|---|---|---|
| `background_sampling.ipynb` | Source/sampler/explainer distance distributions, feature changes, confidence groups | `results_tree`, `dataset`; `proba` for confidence |
| `background_table.ipynb` | Main B exceedance heatmaps and counts | `results_tree` |
| `background_true.ipynb` | Full-background reference distances and exceedance | `results_tree`, `results_true` |
| `background_sweep.ipynb` | Background-size B sweep | `results_tree`, `background_results` |
| `background_topk_table.ipynb` | Top-k set sensitivity | `results_tree` |
| `churn_analysis.ipynb` | Top-1 churn and related rank/set diagnostics | `results_tree` |
| `arrhythmia_analysis.ipynb` | High-dimensional raw event/support analysis | `results_tree` |
| `analysis_full79.py` | Conditional/marginal table on all 79 Diabetes test queries | `results_conditional` |

Run cells in order after reviewing the configuration cells. `background_sampling.ipynb` selects one dataset at a time; change its `DATASET` setting to produce the other datasets. Core B exports go to `figures_b/`. Some core SciencePlots styles also require a LaTeX installation; the appendix plot helpers disable external TeX. The legacy plotting class in `plot_violin.py` is retained for compatibility; use the listed notebooks and `pairwise_aggregation.py` for the paper's B aggregation.

## Appendix figures

The four notebooks in `appendix_e/notebooks/` produce the final appendix presentations. Their helpers are in `analysis/appendix_e/`.

| Notebook | Content |
|---|---|
| `01_churn_depth.ipynb` | Consensus ranks and Top-1 churn |
| `02_topk_depth.ipynb` | Top-k sensitivity heatmap |
| `03_background_size.ipynb` | Background-size Jaccard/RBO plots and compact L2 table |
| `04_arrhythmia.ipynb` | Arrhythmia raw churn and support summaries |

Small count/summary files are included under `figures_b/`. Appendix notebooks 02 and 03 can build their presentations from these included summaries; they do not recompute distances from raw explanations. Notebook 01 reads `results_tree` and `dataset`, and notebook 04 reads `results_tree`, then compares with the included summaries. See [FIGURE_NOTEBOOK_MAP.csv](FIGURE_NOTEBOOK_MAP.csv) and [TABLE_MAP.csv](TABLE_MAP.csv) for notebook cells and paper figure/table numbers.

## Experiment generation

The generators target the original Linux/CUDA research setup. Training and SHAP generation were performed on the NYU Torch GPU server; the Mac workflow analyzes stored results. The analysis environment above is not a replacement for the GPU training environment.

The package supplies model definitions and the `models` import name expected by the preserved generators. This restores source-level imports without changing model parameters, training logic, sampling, or seeds. It does not establish that a fresh run on another machine reproduces historical model weights.

The old prediction generator is retained under `historical/` because its calls do not match the current sampling API. Use the supplied `proba` archive for the published confidence analysis; the package does not claim that this historical generator created every stored probability file. Conditional regeneration also needs the R/conditional-estimator dependencies described with the generation files.

## Interpretation and reproduction scope

The neural sampled/full-background reference comparisons retain the paper's qualification that the same predictive function across those separately generated outputs has not been established. A discrepancy with a reference is therefore not attributed solely to the background for those cases. Within-method repeated-run analyses and their denominators remain distinct.

The conditional comparison uses the matched CPU checkpoints and saved full79 outputs, with different conditional and marginal estimation settings. Its table reports raw Top-1/Top-3 events, not randomized-baseline exceedance or a controlled causal effect of conditioning.

Packaging changes only project paths, import compatibility, environment portability, and notebook presentation. Copied notebook outputs are cleared to avoid presenting historical cell output as a fresh execution. No new training, SHAP generation, or complete notebook run is part of this packaging step. See `docs/PACKAGE_STATUS.md` for the exact validation scope and remaining release inputs.
