
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticRetriever:
    """Retrieve document chunks using sentence embedding similarity."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.chunks: list[dict[str, Any]] = []
        self.chunk_embeddings: np.ndarray | None = None

    def fit(self, chunks: list[dict[str, Any]]) -> None:
        """Encode and index document chunks."""
        if not chunks:
            raise ValueError("Cannot build a retrieval index without chunks.")

        for chunk in chunks:
            if "text" not in chunk or not chunk["text"].strip():
                raise ValueError("Every chunk must contain non-empty text.")

            if "page_number" not in chunk:
                raise ValueError("Every chunk must contain a page_number.")

        texts = [chunk["text"] for chunk in chunks]

        self.chunk_embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        self.chunks = [dict(chunk) for chunk in chunks]

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Return the most semantically similar chunks."""
        if self.chunk_embeddings is None:
            raise RuntimeError("Call fit() before searching.")

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not -1.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between -1 and 1.")

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        # Normalized embeddings allow cosine similarity via dot product.
        scores = self.chunk_embeddings @ query_embedding
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in ranked_indices:
            score = float(scores[index])

            if score < min_score:
                continue

            result = dict(self.chunks[index])
            result["relevance_score"] = score
            results.append(result)

        return results
