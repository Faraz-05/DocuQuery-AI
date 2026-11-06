
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    """Retrieve document chunks relevant to a user's question using TF-IDF."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.chunks: list[dict[str, Any]] = []
        self.chunk_vectors = None

    def fit(self, chunks: list[dict[str, Any]]) -> None:
        """
        Build the TF-IDF index from preprocessed document chunks.
        """
        if not chunks:
            raise ValueError("Cannot build a retrieval index without chunks.")

        texts = []

        for chunk in chunks:
            if "text" not in chunk or not chunk["text"].strip():
                raise ValueError("Every chunk must contain non-empty text.")

            if "page_number" not in chunk:
                raise ValueError("Every chunk must contain a page_number.")

            texts.append(chunk["text"])

        self.chunk_vectors = self.vectorizer.fit_transform(texts)
        self.chunks = [dict(chunk) for chunk in chunks]

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Return the most relevant chunks, ranked by cosine similarity.
        """
        if self.chunk_vectors is None:
            raise RuntimeError("Call fit() before searching.")

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(
            query_vector,
            self.chunk_vectors,
        ).flatten()

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in ranked_indices:
            if scores[index] <= 0:
                continue

            result = dict(self.chunks[index])
            result["relevance_score"] = float(scores[index])
            results.append(result)

        return results
