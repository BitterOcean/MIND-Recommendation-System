from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.metrics import roc_auc_score

from .data import ImpressionSession
from common import ItemCatalog, diversity_at_k, novelty_at_k


class SessionScorer(Protocol):
    def score(self, history: list[str], candidates: list[str]) -> np.ndarray:
        ...


@dataclass(frozen=True)
class EvaluationResult:
    num_sessions: int
    auc: float
    mrr: float
    ndcg_at_5: float
    ndcg_at_10: float
    coverage_at_10: float
    novelty_at_10: float
    diversity_category_at_10: float
    diversity_subcategory_at_10: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "num_sessions": self.num_sessions,
            "auc": self.auc,
            "mrr": self.mrr,
            "ndcg_at_5": self.ndcg_at_5,
            "ndcg_at_10": self.ndcg_at_10,
            "coverage_at_10": self.coverage_at_10,
            "novelty_at_10": self.novelty_at_10,
            "diversity_category_at_10": self.diversity_category_at_10,
            "diversity_subcategory_at_10": self.diversity_subcategory_at_10,
        }


def reciprocal_rank(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="mergesort")
    ranked_labels = labels[order]
    positives = np.flatnonzero(ranked_labels == 1)
    if positives.size == 0:
        return float("nan")
    return float(1.0 / (positives[0] + 1))


def ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    if labels.sum() == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")[:k]
    ranked = labels[order]
    discounts = 1.0 / np.log2(np.arange(2, ranked.size + 2))
    dcg = float(np.sum(ranked * discounts))

    ideal = np.sort(labels)[::-1][:k]
    ideal_dcg = float(np.sum(ideal * discounts[: ideal.size]))
    if ideal_dcg == 0.0:
        return float("nan")
    return dcg / ideal_dcg


def evaluate_ranking_model(
    model: SessionScorer,
    sessions: list[ImpressionSession],
    coverage_k: int = 10,
    catalog: ItemCatalog | None = None,
    beyond_k: int = 10,
) -> EvaluationResult:
    """Evaluate accuracy, coverage, novelty, and diversity.

    Novelty and diversity are computed on the top-``beyond_k`` of each
    session's impression candidates. Pass a catalog built from the training
    split (see ``common.build_item_catalog``) to enable them; omit it to
    get NaNs in those fields.
    """
    auc_scores: list[float] = []
    mrr_scores: list[float] = []
    ndcg5_scores: list[float] = []
    ndcg10_scores: list[float] = []
    novelty_scores: list[float] = []
    diversity_cat_scores: list[float] = []
    diversity_sub_scores: list[float] = []
    recommended_items: set[str] = set()
    candidate_items: set[str] = set()

    for session in sessions:
        scores = model.score(session.history, session.candidates)
        labels = session.labels
        candidate_items.update(session.candidates)

        if np.unique(labels).size > 1:
            auc_scores.append(float(roc_auc_score(labels, scores)))
        if labels.sum() > 0:
            mrr_scores.append(reciprocal_rank(labels, scores))
            ndcg5_scores.append(ndcg_at_k(labels, scores, k=5))
            ndcg10_scores.append(ndcg_at_k(labels, scores, k=10))

        order = np.argsort(-scores, kind="mergesort")
        top_coverage = order[:coverage_k]
        recommended_items.update(session.candidates[idx] for idx in top_coverage)

        if catalog is not None:
            top_beyond = [session.candidates[idx] for idx in order[:beyond_k]]
            novelty_scores.append(novelty_at_k(top_beyond, catalog))
            diversity_cat_scores.append(diversity_at_k(top_beyond, catalog, level="category"))
            diversity_sub_scores.append(diversity_at_k(top_beyond, catalog, level="subcategory"))

    return EvaluationResult(
        num_sessions=len(sessions),
        auc=float(np.nanmean(auc_scores)) if auc_scores else float("nan"),
        mrr=float(np.nanmean(mrr_scores)) if mrr_scores else float("nan"),
        ndcg_at_5=float(np.nanmean(ndcg5_scores)) if ndcg5_scores else float("nan"),
        ndcg_at_10=float(np.nanmean(ndcg10_scores)) if ndcg10_scores else float("nan"),
        coverage_at_10=(
            float(len(recommended_items) / len(candidate_items))
            if candidate_items
            else float("nan")
        ),
        novelty_at_10=float(np.nanmean(novelty_scores)) if novelty_scores else float("nan"),
        diversity_category_at_10=(
            float(np.nanmean(diversity_cat_scores)) if diversity_cat_scores else float("nan")
        ),
        diversity_subcategory_at_10=(
            float(np.nanmean(diversity_sub_scores)) if diversity_sub_scores else float("nan")
        ),
    )