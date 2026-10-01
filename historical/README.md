# Historical probability generator

`experiment_proba.py` is preserved for provenance only and is **not a working generation entry point** for this release. It imports the removed `train_clf_ohe` symbol from `experiment_sampling.py` and calls the older three-argument `make_predict_fn` interface; the current function takes four arguments. Its one-hot preprocessing also belongs to an older generation path.

Use the supplied probability archive for the current confidence analysis. This file is not established as the sole producer of those archived probabilities. It was not rewritten, executed, or used to regenerate probabilities during packaging.
