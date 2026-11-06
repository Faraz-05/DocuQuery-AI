
from pathlib import Path
from typing import Any

from app.nlp.preprocessor import create_chunks
from app.retrieval.semantic_retriever import SemanticRetriever
from app.retrieval.tfidf_retriever import TfidfRetriever
from app.services.document_loader import extract_pdf_text


class DocumentPipeline:
    """Process a PDF and make its text searchable."""

    def __init__(self) -> None:
        self.pages: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []
        self.tfidf_retriever = TfidfRetriever()
        self.semantic_retriever = SemanticRetriever()
        self.document_name: str | None = None
        self.is_ready = False

    def process_pdf(self, file_path: str | Path) -> dict[str, Any]:
        """
        Extract, preprocess, and index a PDF.

        Returns basic document statistics.
        """
        self.is_ready = False
        self.pages = []
        self.chunks = []
        self.document_name = None

        path = Path(file_path)

        pages = extract_pdf_text(path)
        chunks = create_chunks(pages)

        if not chunks:
            raise ValueError("No text chunks could be created from this PDF.")

        # Build both indexes before marking the pipeline as ready.
        self.tfidf_retriever.fit(chunks)
        self.semantic_retriever.fit(chunks)

        self.pages = pages
        self.chunks = chunks
        self.document_name = path.name
        self.is_ready = True

        return {
            "document_name": self.document_name,
            "page_count": len(pages),
            "chunk_count": len(chunks),
            "word_count": sum(
                len(page["text"].split()) for page in pages
            ),
        }

    def search(
        self,
        question: str,
        method: str = "semantic",
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Search indexed chunks using the selected retrieval method."""
        if not self.is_ready:
            raise RuntimeError("Process a PDF before searching.")

        if method == "semantic":
            return self.semantic_retriever.search(question, top_k=top_k)

        if method == "tfidf":
            return self.tfidf_retriever.search(question, top_k=top_k)

        raise ValueError("method must be 'semantic' or 'tfidf'")
