
import hashlib
import sys
import tempfile
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.document_pipeline import DocumentPipeline
from app.services.gemini_service import GeminiAnswerService
from app.qa.qa_service import QAService


st.set_page_config(
    page_title="DocuQuery AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------- Styling --------------------

st.markdown(
    """
    <style>
    .stApp {
        background: #0b1020;
        color: #e5e7eb;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 16, 32, 0.85);
    }

    [data-testid="stSidebar"] {
        background: #10172a;
        border-right: 1px solid #26324b;
    }

    [data-testid="stSidebar"] * {
        color: #dbe4f0;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 1.8rem;
        background: linear-gradient(120deg, #171d38, #24204b, #172e50);
        border: 1px solid #34345f;
        border-radius: 18px;
        margin-bottom: 1.6rem;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.22);
    }

    .hero h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 750;
        margin-bottom: 0.5rem;
    }

    .hero p {
        color: #b9c5e4;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 0;
    }

    .section-heading {
        color: #f1f5f9;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 0.8rem 0;
    }

    [data-testid="stMetric"] {
        background: #121a2d;
        border: 1px solid #293650;
        border-radius: 13px;
        padding: 1rem;
    }

    [data-testid="stMetricLabel"] {
        color: #aab8d0;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    [data-testid="stForm"] {
        background: #121a2d;
        border: 1px solid #293650;
        border-radius: 15px;
        padding: 1.2rem;
    }

    input, textarea {
        background-color: #0d1425 !important;
        color: #f1f5f9 !important;
        border-color: #35425e !important;
    }

    [data-baseweb="select"] > div {
        background-color: #0d1425;
        border-color: #35425e;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        background: linear-gradient(100deg, #6757e8, #4f7cff);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        font-weight: 650;
        min-height: 2.7rem;
        transition: 0.2s ease;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        border: 1px solid #a99cff;
        color: #ffffff;
        filter: brightness(1.12);
    }

    [data-testid="stExpander"] {
        background: #11192b;
        border: 1px solid #293650;
        border-radius: 12px;
    }

    hr {
        border-color: #293650;
    }

    .footer {
        text-align: center;
        color: #8492ad;
        font-size: 0.82rem;
        padding-top: 1.5rem;
    }

    a {
        color: #a99cff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------- Session state --------------------

defaults = {
    "pipeline": None,
    "document_key": None,
    "document_stats": None,
    "last_answer": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def get_field(item, key, default=None):
    """Read a field from a dictionary or an object."""
    if isinstance(item, dict):
        return item.get(key, default)

    return getattr(item, key, default)


def clear_document_state():
    """Clear data belonging to the previously selected PDF."""
    st.session_state.pipeline = None
    st.session_state.document_stats = None
    st.session_state.last_answer = None


def get_page_number(passage):
    """Get a page number while supporting common passage formats."""
    return passage.get(
        "page_number",
        passage.get("page", "Unknown"),
    )


def answer_with_fallback(question, passages):
    """
    Try Gemini first, then extractive QA if Gemini fails.

    QAService's expected input format is question plus context text.
    """
    gemini_error = None

    try:
        gemini_service = GeminiAnswerService()
        result = gemini_service.generate_answer(
            question=question,
            passages=passages,
        )

        answer = str(result.get("answer", "")).strip()

        if answer:
            return {
                "answer": answer,
                "method": result.get("method", "gemini"),
                "sources": result.get("sources", []),
            }

        gemini_error = "Gemini returned an empty answer."

    except Exception as exc:
        gemini_error = f"{type(exc).__name__}: {exc}"

    print(f"[Gemini Fallback] {gemini_error}")

    # Construct context exclusively from the retrieved document passages.
    context_parts = []

    for passage in passages:
        passage_text = str(passage.get("text", "")).strip()

        if passage_text:
            page = get_page_number(passage)
            context_parts.append(f"Page {page}\n{passage_text}")

    context = "\n\n".join(context_parts)

    if not context:
        return {
            "answer": (
                "I couldn't find readable document passages to answer "
                "this question."
            ),
            "method": "abstain",
            "sources": [],
        }

    try:
        qa_service = QAService()
        fallback_result = qa_service.answer(
            question,
            context,
        )

        fallback_answer = str(
            fallback_result.get("answer", "")
        ).strip()

        # Support QAService implementations that return an abstention.
        fallback_method = fallback_result.get("method", "")
        is_abstention = (
            fallback_method == "abstain"
            or not fallback_answer
        )

        if is_abstention:
            return {
                "answer": (
                    fallback_answer
                    or "I couldn't determine the answer from "
                    "the retrieved document excerpts."
                ),
                "method": "abstain",
                "sources": [],
            }

        source_pages = list(
            dict.fromkeys(
                get_page_number(passage)
                for passage in passages
                if str(passage.get("text", "")).strip()
            )
        )

        return {
            "answer": fallback_answer,
            "method": "extractive_fallback",
            "sources": source_pages,
        }

    except Exception as exc:
        print(
            f"[QA Fallback Error] {type(exc).__name__}: {exc}"
        )

        return {
            "answer": (
                "Gemini is temporarily unavailable, and the "
                "extractive fallback could not generate an answer. "
                "Please try again later."
            ),
            "method": "fallback_failed",
            "sources": [],
        }


# -------------------- Sidebar --------------------

with st.sidebar:
    st.markdown("## 📄 DocuQuery AI")
    st.caption("Your document question-answering workspace")
    st.divider()

    st.markdown("### 1. Add a document")

    uploaded_file = st.file_uploader(
        "Upload a text-based PDF",
        type=["pdf"],
        help="Scanned PDFs without extractable text may not work.",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        document_key = hashlib.sha256(file_bytes).hexdigest()

        if st.session_state.document_key != document_key:
            clear_document_state()
            st.session_state.document_key = document_key

        st.caption(f"Selected: {uploaded_file.name}")

        if st.session_state.pipeline is None:
            if st.button(
                "⚙️ Process document",
                type="primary",
                use_container_width=True,
            ):
                temp_path = None

                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf",
                        delete=False,
                    ) as temp_file:
                        temp_file.write(file_bytes)
                        temp_path = Path(temp_file.name)

                    with st.spinner(
                        "Extracting text and building search indexes..."
                    ):
                        pipeline = DocumentPipeline()
                        stats = pipeline.process_pdf(str(temp_path))

                    st.session_state.pipeline = pipeline
                    st.session_state.document_stats = stats
                    st.session_state.last_answer = None

                    st.success("Document is ready.")

                except Exception as exc:
                    clear_document_state()
                    st.error(f"Could not process the PDF: {exc}")

                finally:
                    if temp_path is not None:
                        temp_path.unlink(missing_ok=True)

        else:
            st.success("Document processed and ready to query.")

    else:
        if st.session_state.document_key is not None:
            clear_document_state()
            st.session_state.document_key = None

        st.info("Upload a PDF to begin.")

    st.divider()
    st.markdown("### 2. Search settings")

    retrieval_method = st.selectbox(
        "Retrieval method",
        options=["semantic", "tfidf"],
        format_func=lambda value: (
            "Semantic search"
            if value == "semantic"
            else "TF-IDF keyword search"
        ),
        help="Choose how matching passages are retrieved.",
    )

    top_k = st.slider(
        "Passages to retrieve",
        min_value=3,
        max_value=10,
        value=8,
        help="More passages may help find facts spread across the PDF.",
    )

    st.divider()
    st.caption(
        "Python · Streamlit · NLP · Information retrieval · Gemini AI"
    )


# -------------------- Main header --------------------

st.markdown(
    """
    <div class="hero">
        <h1>Ask your documents.</h1>
        <p>
            Upload a PDF, ask questions in natural language, and get
            structured answers grounded in retrieved document passages.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


pipeline = st.session_state.pipeline
stats = st.session_state.document_stats or {}


# -------------------- Document overview --------------------

if pipeline is not None:
    st.markdown(
        '<div class="section-heading">Document overview</div>',
        unsafe_allow_html=True,
    )

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("Pages", stats.get("page_count", "—"))
    metric2.metric("Text chunks", stats.get("chunk_count", "—"))
    metric3.metric("Words", stats.get("word_count", "—"))

    st.markdown(
        '<div class="section-heading">Ask a question</div>',
        unsafe_allow_html=True,
    )

    with st.form("question_form", clear_on_submit=False):
        question = st.text_input(
            "What would you like to know?",
            placeholder="e.g. Summarize the main obligations in this PDF",
        )

        submitted = st.form_submit_button(
            "🔎 Find answer",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")

        else:
            with st.spinner(
                "Searching the document and generating your answer..."
            ):
                try:
                    results = pipeline.search(
                        question=question.strip(),
                        method=retrieval_method,
                        top_k=top_k,
                    )

                    passages = []

                    for result in results:
                        passage_text = get_field(
                            result,
                            "text",
                            "",
                        )

                        if (
                            not isinstance(passage_text, str)
                            or not passage_text.strip()
                        ):
                            continue

                        passages.append(
                            {
                                "text": passage_text.strip(),
                                "page_number": get_field(
                                    result,
                                    "page_number",
                                    get_field(result, "page", "Unknown"),
                                ),
                            }
                        )

                    if not passages:
                        st.session_state.last_answer = None
                        st.warning(
                            "No readable passages were retrieved. "
                            "Try another search method or rephrase "
                            "your question."
                        )

                    else:
                        answer_result = answer_with_fallback(
                            question.strip(),
                            passages,
                        )

                        st.session_state.last_answer = {
                            "question": question.strip(),
                            "answer": answer_result["answer"],
                            "method": answer_result["method"],
                            "sources": answer_result["sources"],
                            "passages": passages,
                            "retrieval_method": retrieval_method,
                        }

                except Exception as exc:
                    st.session_state.last_answer = None
                    st.error(
                        f"Could not search the document: "
                        f"{type(exc).__name__}: {exc}"
                    )

    # -------------------- Display answer --------------------

    last_answer = st.session_state.last_answer

    if last_answer is not None:
        st.divider()

        st.markdown(
            '<div class="section-heading">Answer</div>',
            unsafe_allow_html=True,
        )

        st.markdown(last_answer["answer"])

        method = last_answer.get("method", "unknown")

        if method == "abstain":
            st.warning(
                "The retrieved passages did not provide enough evidence. "
                "Try increasing the number of passages, switching "
                "retrieval methods, or rephrasing your question."
            )

        elif method == "gemini":
            st.caption(
                "✨ Generated by Gemini using retrieved PDF passages. "
                "Check the source evidence below to verify the answer."
            )

        elif method == "extractive_fallback":
            st.info(
                "Gemini was unavailable. This answer was produced "
                "using extractive question answering and may be less "
                "complete than a generated answer."
            )

        elif method == "fallback_failed":
            st.error(
                "Neither Gemini nor the extractive fallback could "
                "produce an answer."
            )

        else:
            st.caption(f"Answer method: {method}")

        source_pages = last_answer.get("sources", [])

        if source_pages:
            st.markdown("#### 📌 Referenced PDF pages")
            st.write(", ".join(str(page) for page in source_pages))

        st.markdown("#### 📚 Retrieved source evidence")

        for index, passage in enumerate(
            last_answer["passages"],
            start=1,
        ):
            page_number = passage.get("page_number", "Unknown")

            with st.expander(
                f"Passage {index} · PDF page {page_number}",
                expanded=(index == 1),
            ):
                st.write(passage["text"])

        st.caption(
            "The passages shown are retrieval results. Their presence "
            "does not guarantee that every passage supports every part "
            "of the answer."
        )

else:
    st.markdown(
        """
        ### Get started

        1. Upload a text-based PDF from the sidebar.
        2. Select **Process document**.
        3. Enter a question and review the generated answer and evidence.

        Scanned PDFs may require OCR before their contents can be searched.
        """
    )


# -------------------- Footer --------------------

st.markdown(
    """
    <div class="footer">
        DocuQuery AI · Document intelligence · NLP · Information retrieval
    </div>
    """,
    unsafe_allow_html=True,
)
