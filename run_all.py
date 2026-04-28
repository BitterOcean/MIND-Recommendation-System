"""Train and evaluate all four recommenders, then print a comparison table.

Run from the project root:

    python run_all.py
    python run_all.py --max-eval-sessions 5000
    python run_all.py --train-dir data/MINDsmall_train --eval-dir data/MINDsmall_dev
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from baseline.models import PopularityRecommender
from collaborative_filtering.data import build_interaction_dataset
from collaborative_filtering.models import LatentItemCFRecommender
from common import build_item_catalog
from content_based_filtering.models import TfidfContentBasedRecommender
from hybrid_filtering.models import HybridComponent, WeightedHybridRecommender

# Share one evaluator implementation. All three package copies are identical;
# the CF one is arbitrary.
from collaborative_filtering.data import (
    build_eval_sessions,
    read_behaviors,
    read_news,
)
from collaborative_filtering.metrics import evaluate_ranking_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("data/MINDsmall_train"))
    parser.add_argument("--eval-dir", type=Path, default=Path("data/MINDsmall_dev"))
    parser.add_argument("--max-eval-sessions", type=int, default=None)
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
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to write the full results as JSON.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


def format_row(name: str, metrics: dict, train_secs: float, eval_secs: float) -> str:
    def fmt(key: str, width: int = 8) -> str:
        value = metrics.get(key)
        if value is None or (isinstance(value, float) and value != value):  # NaN
            return "  n/a  ".center(width)
        return f"{value:.4f}".rjust(width)

    return (
        f"{name:<10} "
        f"{fmt('auc')} {fmt('mrr')} {fmt('ndcg_at_5')} {fmt('ndcg_at_10')} "
        f"{fmt('diversity_category_at_10')} {fmt('diversity_subcategory_at_10')} "
        f"{fmt('novelty_at_10', 9)} "
        f"{train_secs:>7.1f}s {eval_secs:>7.1f}s"
    )


def print_table(rows: list[tuple[str, dict, float, float]]) -> None:
    header = (
        f"{'Model':<10} "
        f"{'AUC':>8} {'MRR':>8} {'nDCG@5':>8} {'nDCG@10':>8} "
        f"{'Div/cat':>8} {'Div/sub':>8} "
        f"{'Novelty':>9} "
        f"{'Train':>8} {'Eval':>8}"
    )
    print()
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, metrics, train_secs, eval_secs in rows:
        print(format_row(name, metrics, train_secs, eval_secs))
    print("=" * len(header))
    print()
    print("Novelty  = mean -log2 p(item) over top-10  (higher = less popular)")
    print("Div/cat  = unique categories / 10")
    print("Div/sub  = unique subcategories / 10")


def main() -> None:
    args = parse_args()
    train_dir = resolve(args.train_dir)
    eval_dir = resolve(args.eval_dir)

    print(f"Loading data from {train_dir} and {eval_dir} ...")
    train_behaviors = read_behaviors(train_dir / "behaviors.tsv")
    eval_behaviors = read_behaviors(eval_dir / "behaviors.tsv")
    train_news = read_news(train_dir / "news.tsv")
    eval_news = read_news(eval_dir / "news.tsv")
    all_news = pd.concat([train_news, eval_news], ignore_index=True).drop_duplicates("news_id")

    print("Building item catalog (popularity + category metadata) ...")
    catalog = build_item_catalog(train_behaviors, all_news)

    print("Building evaluation sessions ...")
    sessions = build_eval_sessions(eval_behaviors)
    if args.max_eval_sessions is not None:
        sessions = sessions[: args.max_eval_sessions]
    print(f"  {len(sessions)} sessions")

    print("Building interaction matrix for CF ...")
    interactions = build_interaction_dataset(train_behaviors)
    print(
        f"  {len(interactions.user_ids)} users, "
        f"{len(interactions.item_ids)} items, "
        f"{interactions.matrix.nnz} interactions"
    )

    rows: list[tuple[str, dict, float, float]] = []
    results: dict[str, dict] = {}

    def run(name: str, train_fn, model_holder: dict) -> None:
        print(f"\n--- {name} ---")
        t0 = time.perf_counter()
        model = train_fn()
        t1 = time.perf_counter()
        metrics = evaluate_ranking_model(model, sessions, catalog=catalog).to_dict()
        t2 = time.perf_counter()
        train_secs, eval_secs = t1 - t0, t2 - t1
        print(f"  trained in {train_secs:.1f}s, evaluated in {eval_secs:.1f}s")
        rows.append((name, metrics, train_secs, eval_secs))
        results[name] = {"metrics": metrics, "train_secs": train_secs, "eval_secs": eval_secs}
        model_holder["model"] = model

    baseline_holder: dict = {}
    cf_holder: dict = {}
    cbf_holder: dict = {}

    run(
        "Baseline",
        lambda: PopularityRecommender().fit(train_behaviors),
        baseline_holder,
    )
    run(
        "CF",
        lambda: LatentItemCFRecommender(
            n_components=args.cf_components,
            history_decay=args.cf_history_decay,
        ).fit(interactions),
        cf_holder,
    )
    run(
        "CBF",
        lambda: TfidfContentBasedRecommender(
            max_features=args.cbf_max_features,
            min_df=args.cbf_min_df,
            max_df=args.cbf_max_df,
            history_decay=args.cbf_history_decay,
        ).fit(train_news=train_news, all_news=all_news),
        cbf_holder,
    )
    run(
        "Hybrid",
        lambda: WeightedHybridRecommender(
            components=[
                HybridComponent(name="cf", model=cf_holder["model"], weight=args.cf_weight),
                HybridComponent(name="cbf", model=cbf_holder["model"], weight=args.cbf_weight),
            ],
            normalization=args.normalization,
        ),
        {},
    )

    print_table(rows)

    if args.json_output is not None:
        output_path = resolve(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote full results to {output_path}")


if __name__ == "__main__":
    main()