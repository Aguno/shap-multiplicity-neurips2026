# Package scope and verification

Prepared on 2026-10-01 from the current NeurIPS B analysis. This is a code-and-data packaging step, not a fresh experimental reproduction or a completed online release.

## Included and changed

- Seven core notebooks, four appendix notebooks, metric/aggregation helpers, preserved generators, model definitions, environment records, and five small figure-comparison inputs.
- Copied notebook output and execution counters are cleared. Cell positions are preserved so the correspondence tables remain usable.
- Project-root discovery uses `.shap_multiplicity_root`, not a private context document. Appendix notebooks and helpers use their public package locations.
- `models.py` is an exact compatibility copy of the retained `models2.py`. This supplies the module name imported by the generators and referenced in the two CPU checkpoints. Model code, seeds, estimators, and hyperparameters are unchanged.
- The older incompatible probability generator is under `historical/`; use the archived probabilities for confidence analysis.
- Raw explanations, probabilities, datasets and CPU checkpoints are in seven separate ZIPs. The repository contains no raw pickle/checkpoint inputs or result ZIPs.

## Validation performed

- Static Python and notebook-code syntax checks; notebook JSON structure and empty-output checks.
- The five deterministic pairwise-aggregation unit tests pass. The RBO test file is included but its Monte Carlo checks were not run during packaging.
- Portable project-root assignments and the required bundled summary paths were checked.
- Every member of every data ZIP was read back and its SHA256 matched to the original bytes read during packaging. Archive-level hashes and file counts are recorded in `archive_inventory.json`.
- The research originals, model parameters, distance/aggregation formulas, loader precedence, paper PDFs and Overleaf ZIPs were preserved.

These checks do not constitute a full notebook run. No model training, prediction, SHAP computation, baseline simulation, or dependency installation was performed. The currently available local verification environments do not include all plotting dependencies, so complete runtime imports/rendering have not been validated in this new public layout.

## Environment records

`requirements-analysis.txt` lists analysis dependencies using versions present in the saved environment exports. It has not been freshly resolved or installed. `environment.yaml` preserves the compact Python 3.12 / CUDA 11.8 generation specification. `environments/environment_linux_full.yml` preserves the fuller Linux export with the machine-specific prefix removed; it is not a platform-independent lockfile.

Conditional generation additionally needs `pyshapr` and its R dependencies. Their historical versions were not recorded in the supplied environment files, so this package does not invent a pinned installation recipe. Recounting the supplied conditional explanations does not require that generator environment. Checkpoint loading also requires compatible Python, PyTorch and sklearn versions.

Run generators intentionally from the repository root. In particular, `train_cpu_checkpoint.py` trains at module top level, and the conditional generation scripts load data before argument parsing. Importing them or invoking `--help` is not used as a packaging check.

## Online release inputs

The author supplied the OSF archive project, https://osf.io/qa9fk/, and it is recorded in `DATA.md` / `archive_inventory.json`. Public access, upload completeness, and the contents/checksum of any combined ZIP have not been independently verified. The original seven-archive checksums describe the locally verified packaging artifacts. No code license has been chosen by this packaging task. No repository was initialized, committed, pushed, or otherwise modified on GitHub, and the paper's repository link has not been changed.
