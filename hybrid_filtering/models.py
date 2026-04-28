from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from collaborative_filtering.models import LatentItemCFRecommender
from content_based_filtering.models import TfidfContentBasedRecommender


class SessionScorer(Protocol):
    def score(self, history: list[str], candidates: list[str]) -> np.ndarray:
        ...


@dataclass(frozen=True)
class HybridComponent:
    name: str
    model: SessionScorer
    weight: float


class WeightedHybridRecommender:
    def __init__(
        self,
        components: list[HybridComponent],
        normalization: str = "minmax",
    ) -> None:
        if not components:
            raise ValueError("WeightedHybridRecommender requires at least one component.")

        self.components = components
        self.normalization = normalization
        self._validate()

    def score(self, history: list[str], candidates: list[str]) -> np.ndarray:
        if not candidates:
            return np.zeros(0, dtype=np.float32)

        final_scores = np.zeros(len(candidates), dtype=np.float32)
        for component in self.components:
            raw_scores = np.asarray(
                component.model.score(history, candidates),
                dtype=np.float32,
            )
            if raw_scores.shape != final_scores.shape:
                raise ValueError(
                    f"Component '{component.name}' returned scores with shape "
                    f"{raw_scores.shape}, expected {final_scores.shape}."
                )
            final_scores += component.weight * self._normalize_scores(raw_scores)
        return final_scores

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "normalization": self.normalization,
            "components": [],
        }

        for component in self.components:
            component_base = output_path.with_name(f"{output_path.stem}_{component.name}")
            model_path = self._save_component_model(component.model, component_base)
            metadata["components"].append(
                {
                    "name": component.name,
                    "weight": component.weight,
                    "model_type": self._component_model_type(component.model),
                    "model_path": str(model_path.name),
                }
            )

        output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "WeightedHybridRecommender":
        metadata_path = Path(path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        components: list[HybridComponent] = []
        for component_state in metadata["components"]:
            model_path = metadata_path.with_name(component_state["model_path"])
            model_type = component_state["model_type"]

            if model_type == "latent_item_cf":
                model = LatentItemCFRecommender.load(model_path)
            elif model_type == "tfidf_cbf":
                model = TfidfContentBasedRecommender.load(model_path)
            else:
                raise ValueError(f"Unsupported component model type: {model_type}")

            components.append(
                HybridComponent(
                    name=str(component_state["name"]),
                    model=model,
                    weight=float(component_state["weight"]),
                )
            )

        return cls(
            components=components,
            normalization=str(metadata["normalization"]),
        )

    def _validate(self) -> None:
        allowed_normalizations = {"none", "minmax", "zscore", "rank"}
        if self.normalization not in allowed_normalizations:
            raise ValueError(
                f"Unsupported normalization '{self.normalization}'. "
                f"Expected one of {sorted(allowed_normalizations)}."
            )

        total_weight = sum(component.weight for component in self.components)
        if total_weight <= 0.0:
            raise ValueError("The sum of hybrid component weights must be positive.")

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        if scores.size == 0 or self.normalization == "none":
            return scores

        if self.normalization == "minmax":
            min_score = float(scores.min())
            max_score = float(scores.max())
            if max_score == min_score:
                return np.zeros_like(scores)
            return (scores - min_score) / (max_score - min_score)

        if self.normalization == "zscore":
            mean = float(scores.mean())
            std = float(scores.std())
            if std == 0.0:
                return np.zeros_like(scores)
            return (scores - mean) / std

        if self.normalization == "rank":
            order = np.argsort(scores, kind="mergesort")
            ranks = np.empty_like(order, dtype=np.float32)
            ranks[order] = np.arange(scores.size, dtype=np.float32)
            if scores.size == 1:
                return np.ones_like(scores)
            return ranks / float(scores.size - 1)

        raise ValueError(f"Unsupported normalization '{self.normalization}'.")

    def _save_component_model(self, model: SessionScorer, base_path: Path) -> Path:
        if isinstance(model, LatentItemCFRecommender):
            return model.save(base_path.with_suffix(".npz"))
        if isinstance(model, TfidfContentBasedRecommender):
            return model.save(base_path.with_suffix(".json"))
        raise ValueError(f"Unsupported component model instance: {type(model)!r}")

    def _component_model_type(self, model: SessionScorer) -> str:
        if isinstance(model, LatentItemCFRecommender):
            return "latent_item_cf"
        if isinstance(model, TfidfContentBasedRecommender):
            return "tfidf_cbf"
        raise ValueError(f"Unsupported component model instance: {type(model)!r}")
