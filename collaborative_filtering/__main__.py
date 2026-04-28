from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import build_item_catalog

from .data import build_eval_sessions, build_interaction_dataset, read_behaviors, read_news
from .metrics import evaluate_ranking_model
from .models import LatentItemCFRecommender


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a collaborative-filtering baseline on MIND."
    )
    parser.add_argument("--train-dir", type=Path, default=Path("data/MINDsmall_train"))
    parser.add_argument("--eval-dir", type=Path, default=Path("data/MINDsmall_dev"))
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--history-decay", type=float, default=0.9)
    parser.add_argument("--max-eval-sessions", type=int, default=None)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("collaborative_filtering/artifacts/latent_item_cf_model.npz"),
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

    interactions = build_interaction_dataset(train_behaviors)
    sessions = build_eval_sessions(eval_behaviors)
    if args.max_eval_sessions is not None:
        sessions = sessions[: args.max_eval_sessions]

    cf_model = LatentItemCFRecommender(
        n_components=args.components,
        history_decay=args.history_decay,
    ).fit(interactions)
    model_output = resolve_path(args.model_output)
    saved_model_path = cf_model.save(model_output)

    summary = {
        "train": {
            "num_users": len(interactions.user_ids),
            "num_items": len(interactions.item_ids),
            "num_interactions": int(interactions.matrix.nnz),
        },
        "evaluation": {
            "num_sessions": len(sessions),
        },
        "metrics": evaluate_ranking_model(cf_model, sessions, catalog=catalog).to_dict(),
        "cf_model": {
            "components": cf_model.n_components,
            "explained_variance_ratio": cf_model.explained_variance_ratio,
            "history_decay": args.history_decay,
            "model_output": str(saved_model_path),
        },
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()