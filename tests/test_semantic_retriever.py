
import pytest

from app.retrieval.semantic_retriever import SemanticRetriever


@pytest.fixture(scope="module")
def retriever_and_chunks():
    chunks = [
        {
            "chunk_id": 1,
            "page_number": 2,
            "text": "Invoices must be paid within thirty days of receipt.",
        },
        {
            "chunk_id": 2,
            "page_number": 5,
            "text": "Either party may terminate this contract with sixty days written notice.",
        },
        {
            "chunk_id": 3,
            "page_number": 7,
            "text": "The provider's financial liability is capped at the fees paid over twelve months.",
        },
    ]

    retriever = SemanticRetriever()
    retriever.fit(chunks)

    return retriever, chunks


def test_retrieves_semantically_relevant_passage(retriever_and_chunks):
    retriever, _ = retriever_and_chunks

    results = retriever.search(
        "How much advance warning is needed before ending the agreement?",
        top_k=3,
    )

    assert results
    assert results[0]["chunk_id"] == 2
    assert results[0]["page_number"] == 5


def test_results_are_ranked_by_score(retriever_and_chunks):
    retriever, _ = retriever_and_chunks

    results = retriever.search("When must invoices be paid?", top_k=3)

    scores = [result["relevance_score"] for result in results]

    assert scores == sorted(scores, reverse=True)


def test_rejects_search_before_fit():
    retriever = SemanticRetriever()

    with pytest.raises(RuntimeError, match="fit"):
        retriever.search("payment terms")


def test_rejects_empty_query(retriever_and_chunks):
    retriever, _ = retriever_and_chunks

    with pytest.raises(ValueError, match="non-empty"):
        retriever.search("   ")


def test_rejects_empty_chunk_list():
    retriever = SemanticRetriever()

    with pytest.raises(ValueError, match="without chunks"):
        retriever.fit([])
