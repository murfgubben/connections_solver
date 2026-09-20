"""Swappable word embedding interfaces and a gensim Word2Vec implementation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)


class EmbeddingModel(ABC):
    """Minimal interface required by the optimizer."""

    @abstractmethod
    def distance(self, word_a: str, word_b: str) -> float:
        """Return the distance between two words."""

    def distance_matrix(self, words: list[str]) -> np.ndarray:
        matrix = np.zeros((len(words), len(words)), dtype=float)
        for i, word_a in enumerate(words):
            for j in range(i):
                value = self.distance(word_a, words[j])
                matrix[i, j] = matrix[j, i] = value
        return matrix


class Word2VecEmbedding(EmbeddingModel):
    """Google News Word2Vec model with phrase and averaged-token lookup."""

    def __init__(self, model_dir: str | Path = "models") -> None:
        import gensim.downloader as api
        from gensim.models import KeyedVectors

        self.model_dir = Path(model_dir).resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        api.BASE_DIR = str(self.model_dir.resolve())
        LOGGER.info("Loading word2vec-google-news-300 (cached under %s)", self.model_dir)
        # Load the binary directly after using gensim's downloader.  Some gensim
        # versions generate a loader module with a stale absolute cache path;
        # return_path avoids that module while retaining downloader caching.
        model_path = Path(api.load("word2vec-google-news-300", return_path=True))
        if not model_path.is_file():
            raise RuntimeError(f"gensim downloaded model path does not exist: {model_path}")
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        self._vectors: dict[str, np.ndarray | None] = {}

    def _vector(self, word: str) -> np.ndarray | None:
        if word in self._vectors:
            return self._vectors[word]
        candidates = [word.replace(" ", "_"), *word.split()]
        vector = None
        for candidate in candidates[:1]:
            if candidate in self.model.key_to_index:
                vector = np.asarray(self.model[candidate], dtype=float)
                break
        if vector is None and len(candidates) > 1:
            token_vectors = [self.model[token] for token in candidates[1:] if token in self.model.key_to_index]
            if token_vectors:
                vector = np.mean(token_vectors, axis=0)
        if vector is None:
            LOGGER.warning("Word %r is out of vocabulary; using maximum distance", word)
        self._vectors[word] = vector
        return vector

    def distance(self, word_a: str, word_b: str) -> float:
        vector_a, vector_b = self._vector(word_a), self._vector(word_b)
        if vector_a is None or vector_b is None:
            return 1.0
        denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        if denominator == 0:
            return 1.0
        return float(1.0 - np.dot(vector_a, vector_b) / denominator)
