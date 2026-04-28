from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ItemCatalog:
    """Catalog-level metadata used for beyond-accuracy metrics.

    Attributes:
        click_counts: Mapping ``news_id -> positive-click count`` in training
            impressions. Items never clicked are absent (treated as 0).
        total_clicks: Sum of ``click_counts.values()``.
        category: Mapping ``news_id -> category`` from ``news.tsv``.
        subcategory: Mapping ``news_id -> subcategory`` from ``news.tsv``.
    """

    click_counts: dict[str, int]
    total_clicks: int
    category: dict[str, str]
    subcategory: dict[str, str]

    def click_count(self, news_id: str) -> int:
        return self.click_counts.get(news_id, 0)

    def popularity_prob(self, news_id: str) -> float:
        """Add-one smoothed click probability used for novelty."""
        vocab = len(self.click_counts) if self.click_counts else 1
        return (self.click_count(news_id) + 1) / (self.total_clicks + vocab)


def _split_impressions_raw(impressions) -> Iterable[tuple[str, int | None]]:
    if impressions is None or pd.isna(impressions):
        return
    for token in str(impressions).split():
        news_id, sep, label = token.rpartition("-")
        if not sep:
            continue
        try:
            yield news_id, int(label)
        except ValueError:
            continue


def build_item_catalog(
    train_behaviors: pd.DataFrame,
    news: pd.DataFrame,
) -> ItemCatalog:
    """Build an ``ItemCatalog`` from training behaviors and a news table.

    ``train_behaviors`` must have an ``impressions`` column in MIND format.
    ``news`` must have ``news_id``, ``category``, ``subcategory`` columns.
    """
    counts: Counter[str] = Counter()
    total = 0
    for row in train_behaviors.itertuples(index=False):
        for news_id, label in _split_impressions_raw(row.impressions):
            if label == 1:
                counts[news_id] += 1
                total += 1

    news_unique = news.drop_duplicates("news_id")
    category = dict(zip(news_unique["news_id"].astype(str), news_unique["category"].astype(str)))
    subcategory = dict(
        zip(news_unique["news_id"].astype(str), news_unique["subcategory"].astype(str))
    )

    return ItemCatalog(
        click_counts=dict(counts),
        total_clicks=total,
        category=category,
        subcategory=subcategory,
    )
