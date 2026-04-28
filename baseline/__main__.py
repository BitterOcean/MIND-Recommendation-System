from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import build_item_catalog

from .data import build_eval_sessions, read_behaviors, read_news
from .metrics import evaluate_ranking_model
from .models import PopularityRecommender


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a popularity baseline on MIND."
    )
    parser.add_argument("--train-dir", type=Path, default=Path("data/MINDsmall_train"))
    parser.add_argument("--eval-dir", type=Path, default=Path("data/MINDsmall_dev"))
    parser.add_argument("--smoothing", type=float, default=0.0)
    parser.add_argument("--max-eval-sessions", type=int, default=None)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("baseline/artifacts/popularity_model.npz"),
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else Path(__file__).resolve().parent.parent / path


def main() -> None:
    args = parse_args()
    train_dir = resolve_path(args.train_dir)
    eval_dir = resolve_path(args.eval_dir)

    train_behaviors = read_behaviors(train_dir / "behaviors.tsv")
    eval_behaviors = read_behaviors(eval_dir / "behaviors.tsv")
    news = read_news(train_dir / "news.tsv")

    catalog = build_item_catalog(train_behaviors, news)

    sessions = build_eval_sessions(eval_behaviors)
    if args.max_eval_sessions is not None:
        sessions = sessions[: args.max_eval_sessions]

    model = PopularityRecommender(smoothing=args.smoothing).fit(train_behaviors)
    saved_model_path = model.save(resolve_path(args.model_output))

    summary = {
        "train": {
            "num_items_with_clicks": len(model.click_counts),
            "num_positive_clicks": model.total_clicks,
        },
        "evaluation": {
            "num_sessions": len(sessions),
        },
        "metrics": evaluate_ranking_model(model, sessions, catalog=catalog).to_dict(),
        "baseline_model": {
            "type": "popularity",
            "smoothing": args.smoothing,
            "model_output": str(saved_model_path),
        },
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
