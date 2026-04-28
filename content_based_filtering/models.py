from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class NewsFeatureStore:
    news_ids: list[str]
    id_to_index: dict[str, int]
    matrix: sparse.csr_matrix


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def build_content_text(news: pd.DataFrame) -> pd.Series:
    content = (
        news["title"].map(_normalize_text)
        + " "
        + news["abstract"].map(_normalize_text)
        + " "
        + news["category"].map(_normalize_text)
        + " "
        + news["subcategory"].map(_normalize_text)
    )
    return content.str.replace(r"\s+", " ", regex=True).str.strip()


class TfidfContentBasedRecommender:
    def __init__(
        self,
        max_features: int = 50000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        history_decay: float = 0.9,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.history_decay = history_decay

        self.vectorizer: TfidfVectorizer | None = None
        self.news_ids: list[str] = []
        self.news_index: dict[str, int] = {}
        self.news_matrix: sparse.csr_matrix = sparse.csr_matrix((0, 0), dtype=np.float32)

    def fit(
        self,
        train_news: pd.DataFrame,
        all_news: pd.DataFrame | None = None,
    ) -> "TfidfContentBasedRecommender":
        if all_news is None:
            all_news = train_news

        train_news = train_news.drop_duplicates("news_id").reset_index(drop=True)
        all_news = all_news.drop_duplicates("news_id").reset_index(drop=True)

        train_text = build_content_text(train_news)
        all_text = build_content_text(all_news)

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            strip_accents="unicode",
            norm="l2",
            sublinear_tf=True,
        )
        self.vectorizer.fit(train_text)
        self.news_matrix = self.vectorizer.transform(all_text).tocsr().astype(np.float32)
        self.news_ids = all_news["news_id"].astype(str).tolist()
        self.news_index = {news_id: idx for idx, news_id in enumerate(self.news_ids)}
        return self

    def score(self, history: list[str], candidates: list[str]) -> np.ndarray:
        scores = np.zeros(len(candidates), dtype=np.float32)
        if self.vectorizer is None or self.news_matrix.shape[0] == 0:
            return scores

        history_indices = [self.news_index[nid] for nid in history if nid in self.news_index]
        if not history_indices:
            return scores

        history_matrix = self.news_matrix[history_indices]
        weights = self.history_decay ** np.arange(len(history_indices) - 1, -1, -1, dtype=np.float32)
        weights = weights / weights.sum()
        profile = history_matrix.multiply(weights[:, None]).sum(axis=0)
        profile = sparse.csr_matrix(profile)
        profile_norm = np.sqrt(profile.multiply(profile).sum())
        if profile_norm == 0:
            return scores

        candidate_indices = [self.news_index.get(nid, -1) for nid in candidates]
        valid_positions = [pos for pos, idx in enumerate(candidate_indices) if idx >= 0]
        if not valid_positions:
            return scores

        candidate_matrix = self.news_matrix[[candidate_indices[pos] for pos in valid_positions]]
        raw_scores = candidate_matrix @ profile.T
        raw_scores = np.asarray(raw_scores.todense()).reshape(-1)
        for pos, value in zip(valid_positions, raw_scores):
            scores[pos] = float(value)
        return scores

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        matrix_path = output_path.with_suffix(".npz")
        sparse.save_npz(matrix_path, self.news_matrix)

        vectorizer_path = output_path.with_suffix(".vectorizer.pkl")
        with vectorizer_path.open("wb") as f:
            pickle.dump(self.vectorizer, f)

        metadata_path = output_path.with_suffix(".json")
        metadata = {
            "news_ids": self.news_ids,
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
            "min_df": self.min_df,
            "max_df": self.max_df,
            "history_decay": self.history_decay,
            "matrix_path": str(matrix_path.name),
            "vectorizer_path": str(vectorizer_path.name),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return metadata_path

    @classmethod
    def load(cls, path: str | Path) -> "TfidfContentBasedRecommender":
        metadata_path = Path(path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model = cls(
            max_features=int(metadata["max_features"]),
            ngram_range=tuple(metadata["ngram_range"]),
            min_df=int(metadata["min_df"]),
            max_df=float(metadata["max_df"]),
            history_decay=float(metadata["history_decay"]),
        )
        model.news_ids = list(metadata["news_ids"])
        model.news_index = {nid: i for i, nid in enumerate(model.news_ids)}
        model.news_matrix = sparse.load_npz(metadata_path.with_name(metadata["matrix_path"])).tocsr()
        with metadata_path.with_name(metadata["vectorizer_path"]).open("rb") as f:
            model.vectorizer = pickle.load(f)
        return model
