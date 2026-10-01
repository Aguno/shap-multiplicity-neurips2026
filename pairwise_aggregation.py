"""B aggregation for repeated explanations of the same evaluated instances.

Five runs give ten unordered pairs. Threshold each distance first, then divide
by 10*N. This is a comparison frequency, not the fraction of instances passing
an instance-level flag. No model, explainer, baseline, or file is loaded here.
"""

import itertools

import numpy as np


def validate_replicates(values, expected_runs=5):
    """Require complete, finite, equal-shaped (N, F) runs; never trim rows.

    Equal shapes do not establish row identity. Callers must also retain the
    original instance ordering and verify stored row IDs/data where available.
    """
    if len(values) != expected_runs or any(v is None for v in values):
        raise ValueError(f"Expected all {expected_runs} runs; no partial-pair denominator")
    arrays = [np.asarray(v) for v in values]
    shape = arrays[0].shape
    if len(shape) != 2 or min(shape) == 0:
        raise ValueError(f"Expected nonempty (N, F) arrays, got {shape}")
    if any(a.shape != shape for a in arrays):
        raise ValueError("Run shapes differ; refusing to truncate or realign instances")
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("Non-finite attributions; choose an explicit missing-data policy")
    return arrays


def pairwise_distances(values, distance_fn, expected_runs=5):
    """Return (choose(R, 2), N), preserving pair and instance axes."""
    arrays = validate_replicates(values, expected_runs)
    rows = [np.asarray(distance_fn(arrays[i], arrays[j]))
            for i, j in itertools.combinations(range(expected_runs), 2)]
    if any(row.shape != (len(arrays[0]),) for row in rows):
        raise ValueError("Distance function must return exactly one value per instance")
    return np.stack(rows)


def pairwise_exceedance(distances, threshold, expected_pairs=10):
    """Return strict-threshold B percentage and its numerator/denominator.

    The input must retain both axes; flattening and guessing N is not allowed.
    NaNs must not silently count as non-exceedances. No instance mean is taken.
    """
    values = np.asarray(distances)
    if values.ndim != 2 or values.shape[0] != expected_pairs or values.shape[1] == 0:
        raise ValueError(f"Expected ({expected_pairs}, N>0) distances, got {values.shape}")
    if not np.isfinite(values).all() or not np.isfinite(threshold):
        raise ValueError("Distances and threshold must be finite")
    numerator = int(np.count_nonzero(values > threshold))
    denominator = int(values.size)
    return {
        "percent": 100.0 * numerator / denominator,
        "n_exceed": numerator,
        "n_observations": denominator,
        "n_pairs": int(values.shape[0]),
        "n_instances": int(values.shape[1]),
        "threshold": float(threshold),
    }
