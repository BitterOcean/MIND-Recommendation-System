from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BEHAVIOR_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]
NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]


@dataclass(frozen=True)
class ImpressionSession:
    impression_id: int
    user_id: str
    history: list[str]
    candidates: list[str]
    labels: np.ndarray


def read_behaviors(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, names=BEHAVIOR_COLUMNS)


def read_news(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", header=None, names=NEWS_COLUMNS)


def split_history(history: str | float | None) -> list[str]:
    if history is None or pd.isna(history):
        return []
    return [token for token in str(history).split() if token]


def split_impressions(impressions: str | float | None) -> list[tuple[str, int | None]]:
    if impressions is None or pd.isna(impressions):
        return []

    parsed: list[tuple[str, int | None]] = []
    for token in str(impressions).split():
        news_id, separator, label = token.rpartition("-")
        if separator:
            try:
                parsed.append((news_id, int(label)))
            except ValueError:
                parsed.append((token, None))
        else:
            parsed.append((token, None))
    return parsed


def build_eval_sessions(behaviors: pd.DataFrame) -> list[ImpressionSession]:
    sessions: list[ImpressionSession] = []
    for row in behaviors.itertuples(index=False):
        pairs = split_impressions(row.impressions)
        labeled_pairs = [(news_id, label) for news_id, label in pairs if label is not None]
        if not labeled_pairs:
            continue

        candidates = [news_id for news_id, _ in labeled_pairs]
        labels = np.asarray([label for _, label in labeled_pairs], dtype=np.int8)
        sessions.append(
            ImpressionSession(
                impression_id=int(row.impression_id),
                user_id=row.user_id,
                history=split_history(row.history),
                candidates=candidates,
                labels=labels,
            )
        )
    return sessions


def truncate_sessions(
    sessions: Iterable[ImpressionSession], max_sessions: int | None
) -> list[ImpressionSession]:
    if max_sessions is None:
        return list(sessions)
    return list(sessions)[:max_sessions]
