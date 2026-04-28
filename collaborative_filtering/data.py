from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

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
class InteractionDataset:
    user_ids: list[str]
    item_ids: list[str]
    user_index: dict[str, int]
    item_index: dict[str, int]
    matrix: csr_matrix


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


def build_interaction_dataset(
    behaviors: pd.DataFrame,
    include_history: bool = True,
    include_positive_impressions: bool = True,
) -> InteractionDataset:
    per_user_counts: OrderedDict[str, Counter[str]] = OrderedDict()
    observed_items: OrderedDict[str, None] = OrderedDict()

    for row in behaviors.itertuples(index=False):
        user_counts = per_user_counts.setdefault(row.user_id, Counter())

        if include_history:
            for news_id in split_history(row.history):
                user_counts[news_id] += 1
                observed_items.setdefault(news_id, None)

        if include_positive_impressions:
            for news_id, label in split_impressions(row.impressions):
                if label == 1:
                    user_counts[news_id] += 1
                    observed_items.setdefault(news_id, None)

    user_ids = list(per_user_counts.keys())
    item_ids = list(observed_items.keys())
    user_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_index = {item_id: idx for idx, item_id in enumerate(item_ids)}

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []

    for user_id, counts in per_user_counts.items():
        user_idx = user_index[user_id]
        for item_id, value in counts.items():
            item_idx = item_index.get(item_id)
            if item_idx is None:
                continue
            rows.append(user_idx)
            cols.append(item_idx)
            values.append(float(value))

    matrix = csr_matrix(
        (np.asarray(values, dtype=np.float32), (rows, cols)),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float32,
    )

    return InteractionDataset(
        user_ids=user_ids,
        item_ids=item_ids,
        user_index=user_index,
        item_index=item_index,
        matrix=matrix,
    )


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
