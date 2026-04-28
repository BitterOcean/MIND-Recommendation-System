from __future__ import annotations

from typing import Sequence

import numpy as np

from .catalog import ItemCatalog


def novelty_at_k(top_k_items: Sequence[str], catalog: ItemCatalog) -> float:
    """Self-information novelty: mean of ``-log2 p(item)`` over the top-k.

    Higher = less popular items surfaced on average.
    """
    if not top_k_items:
        return float("nan")
    probs = np.fromiter(
        (catalog.popularity_prob(item) for item in top_k_items),
        dtype=np.float64,
        count=len(top_k_items),
    )
    # popularity_prob is add-one smoothed so it's always > 0.
    return float(np.mean(-np.log2(probs)))


def diversity_at_k(
    top_k_items: Sequence[str],
    catalog: ItemCatalog,
    level: str = "category",
) -> float:
    """Fraction of distinct categories in the top-k list.

    ``level`` is either ``"category"`` or ``"subcategory"``. Items missing
    from the catalog's metadata are bucketed as ``"__unknown__"``.
    """
    if not top_k_items:
        return float("nan")
    mapping = catalog.category if level == "category" else catalog.subcategory
    labels = [mapping.get(item, "__unknown__") for item in top_k_items]
    return len(set(labels)) / len(labels)
