from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from collaborative_filtering.data import (
    build_eval_sessions,
    build_interaction_dataset,
    read_behaviors,
)
from collaborative_filtering.metrics import evaluate_ranking_model
from collaborative_filtering.models import LatentItemCFRecommender
from common import build_item_catalog
from content_based_filtering.data import read_news
from content_based_filtering.models import TfidfContentBasedRecommender

from .models import HybridComponent, WeightedHybridRecommender

TUNING_METRICS = ("auc", "mrr", "ndcg_at_5", "ndcg_at_10", "coverage_at_10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a hybrid recommender on MIND."
    )
    parser.add_argument("--train-dir", type=Path, default=Path("data/MINDsmall_train"))
    parser.add_argument("--eval-dir", type=Path, default=Path("data/MINDsmall_dev"))
    parser.add_argument("--cf-components", type=int, default=32)
    parser.add_argument("--cf-history-decay", type=float, default=0.9)
    parser.add_argument("--cbf-history-decay", type=float, default=0.9)
    parser.add_argument("--cbf-max-features", type=int, default=50000)
    parser.add_argument("--cbf-min-df", type=int, default=2)
    parser.add_argument("--cbf-max-df", type=float, default=0.95)
    parser.add_argument("--cf-weight", type=float, default=0.35)
    parser.add_argument("--cbf-weight", type=float, default=0.65)
    parser.add_argument(
        "--normalization",
        choices=["none", "minmax", "zscore", "rank"],
        default="minmax",
    )
    parser.add_argument(
        "--tune-weights",
        action="store_true",
        help="Grid-search cf/cbf weights on the evaluation split.",
    )
    parser.add_argument(
        "--weight-grid-step",
        type=float,
        default=0.1,
        help="Step size for cf weight search in [0, 1].",
    )
    parser.add_argument(
        "--tuning-metric",
        choices=TUNING_METRICS,
        default="ndcg_at_10",
        help="Metric used to select the best weight pair during tuning.",
    )
    parser.add_argument("--max-eval-sessions", type=int, default=None)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("hybrid_filtering/artifacts/weighted_hybrid_model.json"),
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else Path(__file__).resolve().parent.parent / path


def build_hybrid_model(
    cf_model: LatentItemCFRecommender,
    cbf_model: TfidfContentBasedRecommender,
    cf_weight: float,
    cbf_weight: float,
    normalization: str,
) -> WeightedHybridRecommender:
    return WeightedHybridRecommender(
        components=[
            HybridComponent(name="cf", model=cf_model, weight=cf_weight),
            HybridComponent(name="cbf", model=cbf_model, weight=cbf_weight),
        ],
        normalization=normalization,
    )


def build_weight_grid(step: float) -> list[tuple[float, float]]:
    if step <= 0.0 or step > 1.0:
        raise ValueError("--weight-grid-step must be in the interval (0, 1].")

    grid: list[tuple[float, float]] = []
    current = 0.0
    while current < 1.0:
        cf_weight = round(current, 10)
        grid.append((cf_weight, round(1.0 - cf_weight, 10)))
        current += step

    if not grid or grid[-1][0] != 1.0:
        grid.append((1.0, 0.0))
    return grid


def tune_weights(
    cf_model: LatentItemCFRecommender,
    cbf_model: TfidfContentBasedRecommender,
    sessions: list[Any],
    normalization: str,
    step: float,
    target_metric: str,
) -> tuple[float, float, dict[str, float | int], list[dict[str, float | int]]]:
    candidates = build_weight_grid(step)
    best_cf_weight = 0.0
    best_cbf_weight = 1.0
    best_metrics: dict[str, float | int] | None = None
    best_target_score = float("-inf")
    trials: list[dict[str, float | int]] = []

    for cf_weight, cbf_weight in candidates:
        print("Evaluating candidate weights: cf_weight={:.2f}, cbf_weight={:.2f}".format(cf_weight, cbf_weight))
        model = build_hybrid_model(
            cf_model=cf_model,
            cbf_model=cbf_model,
            cf_weight=cf_weight,
            cbf_weight=cbf_weight,
            normalization=normalization,
        )
        metrics = evaluate_ranking_model(model, sessions).to_dict()
        target_score = float(metrics[target_metric])
        trial = {
            "cf_weight": cf_weight,
            "cbf_weight": cbf_weight,
            "target_metric": target_score,
            **metrics,
        }
        trials.append(trial)

        if target_score > best_target_score:
            best_target_score = target_score
            best_cf_weight = cf_weight
            best_cbf_weight = cbf_weight
            best_metrics = metrics

    if best_metrics is None:
        raise RuntimeError("Weight tuning did not evaluate any candidate configurations.")

    return best_cf_weight, best_cbf_weight, best_metrics, trials


def main() -> None:
    args = parse_args()
    train_dir = resolve_path(args.train_dir)
    eval_dir = resolve_path(args.eval_dir)

    train_behaviors = read_behaviors(train_dir / "behaviors.tsv")
    eval_behaviors = read_behaviors(eval_dir / "behaviors.tsv")
    train_news = read_news(train_dir / "news.tsv")
    eval_news = read_news(eval_dir / "news.tsv")

    all_news = pd.concat([train_news, eval_news], ignore_index=True).drop_duplicates("news_id")
    catalog = build_item_catalog(train_behaviors, all_news)

    interactions = build_interaction_dataset(train_behaviors)
    sessions = build_eval_sessions(eval_behaviors)
    if args.max_eval_sessions is not None:
        sessions = sessions[: args.max_eval_sessions]

    cf_model = LatentItemCFRecommender(
        n_components=args.cf_components,
        history_decay=args.cf_history_decay,
    ).fit(interactions)
    cbf_model = TfidfContentBasedRecommender(
        max_features=args.cbf_max_features,
        min_df=args.cbf_min_df,
        max_df=args.cbf_max_df,
        history_decay=args.cbf_history_decay,
    ).fit(train_news=train_news, all_news=all_news)

    tuning_summary = None
    selected_metrics = None
    selected_cf_weight = args.cf_weight
    selected_cbf_weight = args.cbf_weight

    if args.tune_weights:
        (
            selected_cf_weight,
            selected_cbf_weight,
            selected_metrics,
            tuning_trials,
        ) = tune_weights(
            cf_model=cf_model,
            cbf_model=cbf_model,
            sessions=sessions,
            normalization=args.normalization,
            step=args.weight_grid_step,
            target_metric=args.tuning_metric,
        )
        tuning_summary = {
            "enabled": True,
            "target_metric": args.tuning_metric,
            "weight_grid_step": args.weight_grid_step,
            "num_trials": len(tuning_trials),
            "best_cf_weight": selected_cf_weight,
            "best_cbf_weight": selected_cbf_weight,
            "trials": tuning_trials,
        }

    hybrid_model = build_hybrid_model(
        cf_model=cf_model,
        cbf_model=cbf_model,
        cf_weight=selected_cf_weight,
        cbf_weight=selected_cbf_weight,
        normalization=args.normalization,
    )
    saved_model_path = hybrid_model.save(resolve_path(args.model_output))
    metrics = selected_metrics or evaluate_ranking_model(hybrid_model, sessions).to_dict()

    summary = {
        "train": {
            "num_users": len(interactions.user_ids),
            "num_items": len(interactions.item_ids),
            "num_interactions": int(interactions.matrix.nnz),
            "num_train_news": int(train_news["news_id"].nunique()),
            "num_eval_news": int(eval_news["news_id"].nunique()),
            "num_total_news_indexed": len(cbf_model.news_ids),
        },
        "evaluation": {
            "num_sessions": len(sessions),
        },
        "metrics": metrics,
        "hybrid_model": {
            "normalization": args.normalization,
            "cf_weight": selected_cf_weight,
            "cbf_weight": selected_cbf_weight,
            "model_output": str(saved_model_path),
        },
        "components": {
            "cf": {
                "components": cf_model.n_components,
                "explained_variance_ratio": cf_model.explained_variance_ratio,
                "history_decay": args.cf_history_decay,
            },
            "cbf": {
                "type": "tfidf_user_profile",
                "history_decay": args.cbf_history_decay,
                "max_features": args.cbf_max_features,
                "min_df": args.cbf_min_df,
                "max_df": args.cbf_max_df,
                "feature_dim": int(cbf_model.news_matrix.shape[1]),
            },
        },
    }
    if tuning_summary is not None:
        summary["weight_tuning"] = tuning_summary

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()