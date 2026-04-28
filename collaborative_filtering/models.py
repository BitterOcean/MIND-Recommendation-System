from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD

from .data import InteractionDataset


class LatentItemCFRecommender:
    def __init__(
        self,
        n_components: int = 32,
        history_decay: float = 0.9,
        random_state: int = 42,
    ) -> None:
        self.requested_components = n_components
        self.history_decay = history_decay
        self.random_state = random_state
        self.item_ids: list[str] = []
        self.item_index: dict[str, int] = {}
        self.item_embeddings = np.zeros((0, 0), dtype=np.float32)
        self.n_components = 0
        self.explained_variance_ratio = 0.0

    def fit(self, interactions: InteractionDataset) -> "LatentItemCFRecommender":
        self.item_ids = interactions.item_ids
        self.item_index = interactions.item_index

        item_user_matrix = interactions.matrix.T.tocsr()
        max_components = min(item_user_matrix.shape) - 1
        self.n_components = max(1, min(self.requested_components, max_components))

        svd = TruncatedSVD(
            n_components=self.n_components,
            algorithm="randomized",
            n_iter=7,
            random_state=self.random_state,
        )
        embeddings = svd.fit_transform(item_user_matrix).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self.item_embeddings = embeddings / norms
        self.explained_variance_ratio = float(svd.explained_variance_ratio_.sum())
        return self

    def score(self, history: list[str], candidates: list[str]) -> np.ndarray:
        return self._score_candidates_from_profile(history, candidates)

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            item_ids=np.asarray(self.item_ids, dtype=str),
            item_embeddings=self.item_embeddings,
            requested_components=np.asarray(self.requested_components, dtype=np.int32),
            n_components=np.asarray(self.n_components, dtype=np.int32),
            history_decay=np.asarray(self.history_decay, dtype=np.float32),
            random_state=np.asarray(self.random_state, dtype=np.int32),
            explained_variance_ratio=np.asarray(
                self.explained_variance_ratio, dtype=np.float32
            ),
        )
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "LatentItemCFRecommender":
        model_path = Path(path)
        with np.load(model_path, allow_pickle=False) as state:
            model = cls(
                n_components=int(state["requested_components"]),
                history_decay=float(state["history_decay"]),
                random_state=int(state["random_state"]),
            )
            model.item_ids = state["item_ids"].tolist()
            model.item_index = {item_id: idx for idx, item_id in enumerate(model.item_ids)}
            model.item_embeddings = state["item_embeddings"].astype(np.float32)
            model.n_components = int(state["n_components"])
            model.explained_variance_ratio = float(state["explained_variance_ratio"])
        return model

    def _score_candidates_from_profile(
        self, history: list[str], candidates: list[str]
    ) -> np.ndarray:
        candidate_scores = np.zeros(len(candidates), dtype=np.float32)
        history_indices = [self.item_index[item] for item in history if item in self.item_index]
        if not history_indices:
            return candidate_scores

        history_vectors = self.item_embeddings[history_indices]
        weights = self.history_decay ** np.arange(len(history_indices) - 1, -1, -1, dtype=np.float32)
        profile = np.average(history_vectors, axis=0, weights=weights)
        profile_norm = float(np.linalg.norm(profile))
        if profile_norm == 0.0:
            return candidate_scores
        profile /= profile_norm

        for idx, candidate in enumerate(candidates):
            item_idx = self.item_index.get(candidate)
            if item_idx is not None:
                candidate_scores[idx] = float(np.dot(profile, self.item_embeddings[item_idx]))

        return candidate_scores
