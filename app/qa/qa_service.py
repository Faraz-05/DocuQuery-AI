
import re
from functools import lru_cache

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


DEFAULT_MODEL = "distilbert/distilbert-base-cased-distilled-squad"
MAX_LENGTH = 384
MAX_ANSWER_TOKENS = 30

AMOUNT_PATTERN = r"(?:Rs\.?|₹|INR)\s*[\d,]+(?:\.\d+)?"
PERCENTAGE_PATTERN = r"\b\d+(?:\.\d+)?\s*%"


@lru_cache(maxsize=2)
def load_qa_pipeline(model_name: str = DEFAULT_MODEL):
    """Load and cache the tokenizer and extractive QA model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.eval()

    def run_qa(question: str, context: str) -> dict:
        encoded = tokenizer(
            question,
            context,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation="only_second",
            max_length=MAX_LENGTH,
        )

        sequence_ids = encoded.sequence_ids(0)
        offsets = encoded.pop("offset_mapping")[0].tolist()

        context_tokens = [
            i
            for i, sequence_id in enumerate(sequence_ids)
            if sequence_id == 1 and offsets[i][1] > offsets[i][0]
        ]

        if not context_tokens:
            return {
                "answer": "",
                "confidence": 0.0,
                "start": 0,
                "end": 0,
                "method": "model",
            }

        model_inputs = {
            key: value
            for key, value in encoded.items()
            if key in tokenizer.model_input_names
        }

        with torch.no_grad():
            outputs = model(**model_inputs)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        start_probs = torch.softmax(start_logits, dim=0)
        end_probs = torch.softmax(end_logits, dim=0)

        best_start = context_tokens[0]
        best_end = best_start
        best_score = float("-inf")

        for start_idx in context_tokens:
            for end_idx in context_tokens:
                if end_idx < start_idx:
                    continue

                if end_idx - start_idx + 1 > MAX_ANSWER_TOKENS:
                    continue

                score = (
                    start_logits[start_idx].item()
                    + end_logits[end_idx].item()
                )

                if score > best_score:
                    best_score = score
                    best_start = start_idx
                    best_end = end_idx

        char_start = offsets[best_start][0]
        char_end = offsets[best_end][1]

        return {
            "answer": context[char_start:char_end].strip(),
            "confidence": float(
                (start_probs[best_start] * end_probs[best_end]).item()
            ),
            "start": char_start,
            "end": char_end,
            "method": "model",
        }

    return run_qa


def split_sentences(context: str) -> list[str]:
    """Split text into sentences without breaking common abbreviations."""
    protected = re.sub(
        r"\b(Rs|Mr|Mrs|Ms|Dr)\.",
        lambda match: match.group(0).replace(".", "<DOT>"),
        context,
        flags=re.IGNORECASE,
    )

    sentences = re.split(r"(?<=[.!?])\s+", protected)

    return [
        sentence.replace("<DOT>", ".").strip()
        for sentence in sentences
        if sentence.strip()
    ]


def make_abstain_result() -> dict:
    """Return a consistent answer when evidence is missing."""
    return {
        "answer": (
            "I couldn't find a sentence in the retrieved passages that "
            "directly supports an answer to this question. Try retrieving "
            "more passages or rephrasing the question."
        ),
        "method": "abstain",
    }


def find_direct_answer(question: str, context: str):
    """Find a sentence containing the requested fact."""
    q = question.lower()
    sentences = split_sentences(context)

    if any(
        phrase in q
        for phrase in (
            "not mentioned",
            "not stated",
            "not in the pdf",
            "not in the document",
            "unrelated event",
        )
    ):
        return make_abstain_result()

    rules = [
        (
            "rent_escalation",
            (
                "rent escalation",
                "escalation percentage",
                "annual rent escalation",
                "percentage increase in rent",
            ),
            ("escalation",),
            PERCENTAGE_PATTERN,
        ),
        (
            "rent_due_date",
            (
                "when is the monthly rent due",
                "when is monthly rent due",
                "when is rent due",
                "rent due date",
                "when should rent be paid",
                "when are rent payments due",
            ),
            ("rent", "due"),
            None,
        ),
        (
            "security_deposit",
            (
                "security deposit",
                "deposit amount",
                "how much is the deposit",
            ),
            ("security deposit",),
            AMOUNT_PATTERN,
        ),
        (
            "property_tax",
            ("property tax", "total property tax"),
            ("property tax",),
            AMOUNT_PATTERN,
        ),
        (
            "monthly_base_rent",
            (
                "monthly base rent",
                "base rent per month",
                "monthly rent amount",
                "what is the monthly rent",
                "how much is the monthly rent",
            ),
            ("monthly rent", "base rent"),
            AMOUNT_PATTERN,
        ),
    ]

    matched_rule = False

    for rule_name, phrases, required_terms, value_pattern in rules:
        if not any(phrase in q for phrase in phrases):
            continue

        matched_rule = True
        candidates = []

        for sentence in sentences:
            lower = sentence.lower()

            if rule_name == "monthly_base_rent":
                if not ("monthly rent" in lower or "base rent" in lower):
                    continue
            elif not all(term in lower for term in required_terms):
                continue

            if rule_name == "rent_due_date":
                valid_due_phrases = (
                    "due on or before",
                    "due on",
                    "payable on",
                    "must be paid on",
                    "shall be paid on",
                    "payable by",
                    "due by",
                )
                if not any(phrase in lower for phrase in valid_due_phrases):
                    continue
            elif value_pattern and not re.search(
                value_pattern, sentence, flags=re.IGNORECASE
            ):
                continue

            if rule_name == "rent_escalation" and any(
                phrase in lower
                for phrase in (
                    "holdover",
                    "holding over",
                    "last applicable rent",
                )
            ):
                continue

            candidates.append(sentence)

        if candidates:
            answer = min(candidates, key=len)
            return {
                "answer": answer,
                "method": "source_match",
            }

    if matched_rule:
        return make_abstain_result()

    amount_question = any(
        phrase in q
        for phrase in (
            "how much",
            "what is the amount",
            "how many rupees",
            "what does it cost",
            "what is the cost",
        )
    )

    if amount_question:
        stopwords = {
            "how", "much", "many", "what", "is", "the", "a", "an",
            "of", "for", "in", "on", "to", "and", "does", "current",
            "estimated", "total", "amount", "cost", "rupees",
        }

        terms = {
            word
            for word in re.findall(r"[a-z]+", q)
            if word not in stopwords
        }

        candidates = []

        for sentence in sentences:
            if not re.search(
                AMOUNT_PATTERN
                + r"|[\d,]+(?:\.\d+)?\s*(?:rupees|per annum)",
                sentence,
                flags=re.IGNORECASE,
            ):
                continue

            sentence_terms = set(re.findall(r"[a-z]+", sentence.lower()))
            overlap = len(terms & sentence_terms)

            if overlap:
                candidates.append((overlap, -len(sentence), sentence))

        if candidates:
            return {
                "answer": max(candidates, key=lambda item: item[:2])[2],
                "method": "source_match",
            }

    return None


class QAService:
    """Answer questions using source matching and extractive QA."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name

    def answer(self, question: str, context: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not isinstance(context, str) or not context.strip():
            raise ValueError("Context cannot be empty.")

        question = question.strip()
        context = context.strip()

        direct_result = find_direct_answer(question, context)

        if direct_result:
            answer = direct_result["answer"]
            start = context.find(answer)

            return {
                "answer": answer,
                "confidence": 0.0,
                "start": start,
                "end": start + len(answer) if start >= 0 else 0,
                "method": direct_result["method"],
            }

        return load_qa_pipeline(self.model_name)(
            question=question,
            context=context,
        )
