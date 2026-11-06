
import pytest

from app.retrieval.tfidf_retriever import TfidfRetriever


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": 1,
            "page_number": 2,
            "text": "The client must pay invoices within 30 days.",
        },
        {
            "chunk_id": 2,
            "page_number": 5,
            "text": "Either party may terminate the agreement with 60 days written notice.",
        },
        {
            "chunk_id": 3,
            "page_number": 7,
            "text": "The provider's liability is limited to fees paid during 12 months.",
        },
    ]


def test_retrieves_relevant_chunk(sample_chunks):
    retriever = TfidfRetriever()
    retriever.fit(sample_chunks)

    results = retriever.search(
        "How much notice is required to terminate the agreement?",
        top_k=2,
    )

    assert results
    assert results[0]["chunk_id"] == 2
    assert results[0]["page_number"] == 5
    assert results[0]["relevance_score"] > 0


def test_results_are_ranked_by_relevance(sample_chunks):
    retriever = TfidfRetriever()
    retriever.fit(sample_chunks)

    results = retriever.search(
        "pay invoices within 30 days",
        top_k=3,
    )

    assert results[0]["chunk_id"] == 1
    scores = [result["relevance_score"] for result in results]
    assert scores == sorted(scores, reverse=True)


def test_rejects_search_before_fit():
    retriever = TfidfRetriever()

    with pytest.raises(RuntimeError, match="fit"):
        retriever.search("payment terms")


def test_rejects_empty_query(sample_chunks):
    retriever = TfidfRetriever()
    retriever.fit(sample_chunks)

    with pytest.raises(ValueError, match="non-empty"):
        retriever.search("   ")


def test_rejects_empty_chunk_list():
    retriever = TfidfRetriever()

    with pytest.raises(ValueError, match="without chunks"):
        retriever.fit([])


def test_returns_no_results_for_unrelated_query(sample_chunks):
    retriever = TfidfRetriever()
    retriever.fit(sample_chunks)

    results = retriever.search(
        "astronomy and distant galaxies",
        top_k=3,
    )

    assert results == []
