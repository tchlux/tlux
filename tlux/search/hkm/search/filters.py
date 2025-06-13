"""Query-time filtering helpers.

The helper :func:`apply_filters` trims a candidate *docs* iterable to the
subset satisfying the *label* and *numeric* constraints contained in a
:class:`~hkm.schema.QuerySpec` (or compatible mapping).

Document objects may be either **dict-like** or simple objects with two
attributes:

* ``labels``  - mapping ``str -> (str | list[str])``
* ``numeric`` - mapping ``str -> float``

Any document lacking a required field is *rejected*.
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence

from ..schema import QuerySpec

__all__ = ["apply_filters"]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _get(doc, key: str, default=None):
    """Retrieve *key* from *doc* regardless of it being a dict or object."""
    if isinstance(doc, Mapping):
        return doc.get(key, default)
    return getattr(doc, key, default)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_filters(docs: Iterable, query: QuerySpec) -> List:
    """Return documents satisfying *query* label & numeric constraints.

    Parameters
    ----------
    docs:
        Iterable of candidate documents (dict-like or objects).  Each
        must expose ``labels`` and ``numeric`` as described above.
    query:
        The parsed :class:`QuerySpec` containing *label_include* and
        *numeric_range* conditions.  Other fields are ignored here.

    Returns
    -------
    selected:
        List of documents that meet **all** constraints.
    """

    label_req: Mapping[str, Sequence[str]] = query.label_include or {}
    numeric_req: Mapping[str, Sequence[float]] = query.numeric_range or {}

    selected = []
    for doc in docs:
        labels = _get(doc, "labels", {})
        numeric = _get(doc, "numeric", {})

        # -- Label filters ------------------------------------------------
        ok = True
        for key, allowed in label_req.items():
            val = labels.get(key)
            if val is None:
                ok = False
                break
            if isinstance(val, (list, tuple, set)):
                if not any(x in allowed for x in val):
                    ok = False
                    break
            else:
                if val not in allowed:
                    ok = False
                    break
        if not ok:
            continue

        # -- Numeric filters ---------------------------------------------
        for key, bounds in numeric_req.items():
            if len(bounds) != 2:
                raise ValueError(
                    f"numeric_range for '{key}' must be a (low, high) pair"
                )
            val = numeric.get(key)
            if val is None or not (bounds[0] <= val <= bounds[1]):
                ok = False
                break
        if ok:
            selected.append(doc)

    return selected
