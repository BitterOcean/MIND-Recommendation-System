from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from common import build_item_catalog

from .data import build_eval_sessions, read_behaviors, read_news
from .metrics import evaluate_ranking_model
from .models import TfidfContentBasedRecommender


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a content-based filtering baseline on MIND."
    )
    parser.add_argument("--train-dir", type=Path, default=Path("data/MINDsmall_train"))
    parser.add_argument("--eval-dir", type=Path, default=Path("data/MINDsmall_dev"))
    parser.add_argument("--history-decay", type=float, default=0.9)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--max-eval-sessions", type=int, default=None)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("content_based_filtering/artifacts/tfidf_cbf_model.json"),
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else Path(__file__).resolve().parent.parent / path


def main() -> None:
    args = parse_args()
    train_dir = resolve_path(args.train_dir)
    eval_dir = resolve_path(args.eval_dir)

    train_news = read_news(train_dir / "news.tsv")
    eval_news = read_news(eval_dir / "news.tsv")
    all_news = pd.concat([train_news, eval_news], ignore_index=True).drop_duplicates("news_id")

    train_behaviors = read_behaviors(train_dir / "behaviors.tsv")
    eval_behaviors = read_behaviors(eval_dir / "behaviors.tsv")

    catalog = build_item_catalog(train_behaviors, all_news)

    sessions = build_eval_sessions(eval_behaviors)
    if args.max_eval_sessions is not None:
        sessions = sessions[: args.max_eval_sessions]

    model = TfidfContentBasedRecommender(
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        history_decay=args.history_decay,
    ).fit(train_news=train_news, all_news=all_news)

    saved_model_path = model.save(resolve_path(args.model_output))

    summary = {
        "train": {
            "num_train_news": int(train_news["news_id"].nunique()),
            "num_eval_news": int(eval_news["news_id"].nunique()),
            "num_total_news_indexed": len(model.news_ids),
            "feature_dim": int(model.news_matrix.shape[1]),
        },
        "evaluation": {
            "num_sessions": len(sessions),
        },
        "metrics": evaluate_ranking_model(model, sessions, catalog=catalog).to_dict(),
        "cbf_model": {
            "type": "tfidf_user_profile",
            "history_decay": args.history_decay,
            "max_features": args.max_features,
            "min_df": args.min_df,
            "max_df": args.max_df,
            "model_output": str(saved_model_path),
        },
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()