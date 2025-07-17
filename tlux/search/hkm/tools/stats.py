"""Label and numeric aggregation helpers for HKM nodes.

The output structure follows the JSON schema sketched in the README:

    {
        "labels_count": {"label0": 123, ...},
        "numeric_min":  [min0, min1, ...],
        "numeric_max":  [max0, max1, ...],
        "numeric_hist": {"feat": [[lo, hi, cnt], ...]},
    }

The helper uses NumPy for efficient computation but works with any
*array-like* inputs (e.g. Python lists).
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

__all__ = ["compute_stats"]


def _compute_numeric_hist(
    values: np.ndarray, n_bins: int | None = None
) -> List[List[float]]:
    """Return histogram triples ``[lo, hi, count]``.

    If *n_bins* is *None*, the **Freedman-Diaconis** rule is used to pick
    an adaptive bin width that balances resolution and robustness.
    """

    if values.size == 0:
        return []

    if n_bins is None:
        q75, q25 = np.percentile(values, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            n_bins = 1  # all identical
        else:
            bin_width = 2 * iqr / (values.size ** (1 / 3))
            n_bins = max(1, int(np.ceil((values.max() - values.min()) / bin_width)))

    counts, edges = np.histogram(values, bins=n_bins)
    return [[float(edges[i]), float(edges[i + 1]), int(counts[i])] for i in range(len(counts))]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_stats(
    labels: Iterable[str] | Sequence[str],
    numeric: Mapping[str, Sequence[float] | np.ndarray],
) -> Dict[str, object]:
    """Aggregate label counts and numeric summaries for a node.

    Parameters
    ----------
    labels:
        Iterable of *categorical* labels (strings).  Multiple labels per
        document should be flattened beforehand.
    numeric:
        Mapping from feature name to sequence/array of numeric values.

    Returns
    -------
    stats:
        Dictionary ready for ``json.dumps`` matching the HKM spec.
    """

    # --- Labels ----------------------------------------------------------
    label_counter = Counter(labels)

    # --- Numeric features ------------------------------------------------
    numeric_min: List[float] = []
    numeric_max: List[float] = []
    numeric_hist: Dict[str, List[List[float]]] = {}

    for feat, vals in numeric.items():
        arr = np.asarray(list(vals), dtype=np.float32)
        if arr.size == 0:
            numeric_min.append(float("nan"))
            numeric_max.append(float("nan"))
            numeric_hist[feat] = []
            continue

        numeric_min.append(float(arr.min()))
        numeric_max.append(float(arr.max()))
        numeric_hist[feat] = _compute_numeric_hist(arr)

    return {
        "labels_count": dict(label_counter),
        "numeric_min": numeric_min,
        "numeric_max": numeric_max,
        "numeric_hist": numeric_hist,
    }
