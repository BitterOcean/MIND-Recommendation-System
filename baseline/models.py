from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .data import split_impressions


class PopularityRecommender:
    """Popularity baseline.

    Scores each candidate by the number of positive clicks it received in the
    training impressions. Exposes the same ``score(history, candidates)``
    interface as the CF and CBF models so it can be evaluated with the shared
    ``evaluate_ranking_model`` and plugged into ``WeightedHybridRecommender``
    as an additional component.
    """

    def __init__(self, smoothing: float = 0.0) -> None:
        self.smoothing = float(smoothing)
        self.click_counts: dict[str, int] = {}
        self.total_clicks: int = 0

    def fit(self, behaviors: pd.DataFrame) -> "PopularityRecommender":
        counts: Counter[str] = Counter()
        total = 0
        for row in behaviors.itertuples(index=False):
            for news_id, label in split_impressions(row.impressions):
                if label == 1:
                    counts[news_id] += 1
                    total += 1
        self.click_counts = dict(counts)
        self.total_clicks = total
        return self

    def score(self, history: list[str], candidates: list[str]) -> np.ndarray:
        # History is intentionally ignored: popularity is user-independent.
        del history
        scores = np.empty(len(candidates), dtype=np.float32)
        for i, news_id in enumerate(candidates):
            scores[i] = self.click_counts.get(news_id, 0) + self.smoothing
        return scores

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        news_ids = np.asarray(list(self.click_counts.keys()), dtype=str)
        counts = np.asarray(list(self.click_counts.values()), dtype=np.int64)
        np.savez_compressed(
            output_path,
            news_ids=news_ids,
            click_counts=counts,
            total_clicks=np.asarray(self.total_clicks, dtype=np.int64),
            smoothing=np.asarray(self.smoothing, dtype=np.float32),
        )
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "PopularityRecommender":
        model_path = Path(path)
        with np.load(model_path, allow_pickle=False) as state:
            model = cls(smoothing=float(state["smoothing"]))
            news_ids = state["news_ids"].tolist()
            click_counts = state["click_counts"].tolist()
            model.click_counts = {
                news_id: int(count) for news_id, count in zip(news_ids, click_counts)
            }
            model.total_clicks = int(state["total_clicks"])
        return model
