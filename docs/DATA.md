# Data and result archives

Large and binary research inputs are separate from the code repository. The archive project is [OSF](https://osf.io/qa9fk/). Extract the result directories into the code repository root, without adding an extra enclosing directory.

The planned combined archive contains these seven top-level folders: `results_tree/`, `results_true/`, `background_results/`, `results_conditional/`, `proba/`, `dataset/`, and `models_cpu/`. Preserve `proba/proba/` inside that layout. Public access and the uploaded files have not been independently verified during this documentation update.

If downloaded as the combined ZIP named below, extract it from the code repository root:

```bash
unzip /path/to/downloads/shap-multiplicity-neurips2026-data-all.zip -d .
```

Use the actual downloaded filename if different. The table below records the seven separately prepared archives and their contents; it is not a verified listing of the files currently uploaded to OSF.

| Archive | Contents | Used for |
|---|---|---|
| `results_tree.zip` | Stored main and Arrhythmia explanation outputs | Main B, source/method comparison, churn, Top-k, Arrhythmia |
| `results_true.zip` | Full-background reference explanations | Reference plots and tables |
| `background_results.zip` | Stored background-size sweep outputs | K sweep beyond the reused K100 inputs |
| `results_conditional.zip` | Twenty conditional/marginal full79 explanation files | Conditional appendix table |
| `proba.zip` | Stored predictions, preserving `proba/proba/` | Confidence-group plots |
| `dataset.zip` | Ten saved dataset input/target files | Feature labels, query-alignment checks, experiment input |
| `models_cpu.zip` | Two matched CPU checkpoints | Conditional/marginal regeneration; not required merely to count saved explanations |

**Archive project:** [OSF](https://osf.io/qa9fk/). The file `docs/archive_inventory.json` records file counts, byte sizes, and SHA256 checksums. The accompanying archive distribution also provides per-file checksum manifests. A download link alone should not replace these version identifiers.

The inventory hashes refer to the seven individual ZIPs, not to a newly combined ZIP. If individual ZIPs are supplied, extract them from the code repository root, for example:

```bash
unzip /path/to/downloads/results_tree.zip -d .
unzip /path/to/downloads/dataset.zip -d .
unzip /path/to/downloads/results_true.zip -d .
```

The resulting paths should be `results_tree/...`, `dataset/...`, and `results_true/...`. For confidence plots the expected stored path is `proba/proba/...`; keep that nested layout.

The archives preserve the existing folders without changing explanation arrays. A folder can contain both integrated and chunk versions. The public loaders preserve the paper's selected input precedence: K100 whole files for German/Diabetes and complete chunks for ACS/GMSC. Do not merge or rename those alternatives on the assumption that they contain identical explanations.

Not every file in the archived folders is used by every figure. K3200 or other historical settings, if present, are not additional required experiments. The current paper's sweep ends at K1600 or the available training size. The excluded legacy `shap_values`/coalition/leverage folders are not part of this data distribution.

Dataset redistribution and third-party licenses remain those of their original sources; this package does not assign a new license to them. Direct file download URLs, the combined-archive checksum, and the repository's code license can be added when the authors finalize publication.
