from typing import Dict, List

from .text_cleaner import clean_text_for_web
from .header_detector import smart_clean_document
from .nlp_utils import build_tfidf_matrix, rank_sentences, split_sentences


def _select_sentence_indices(
    sentences: List[str],
    ratio: float,
) -> List[int]:
    """Return indices of sentences selected for the summary."""
    if not sentences:
        return []

    try:
        _, matrix = build_tfidf_matrix(sentences)
        scores: Dict[int, float] = rank_sentences(sentences, matrix)
    except ValueError:
        # Fallback when TF-IDF cannot be computed (e.g. empty sentences)
        scores = {idx: float(len(sent.split())) for idx, sent in enumerate(sentences)}

    nb_sentences = max(1, int(len(sentences) * ratio))
    ranked_indices = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_indices = [idx for idx, _ in ranked_indices[:nb_sentences]]

    # Preserve original order for readability
    return sorted(top_indices)


def generate_summary(text: str, length: str = "medium") -> str:
    """Generate an extractive summary with TF-IDF enhanced scoring."""
    text = clean_text_for_web(text)
    text = smart_clean_document(text)

    sentences = split_sentences(text)
    if not sentences:
        return clean_text_for_web(text[:300])

    ratio_map = {"short": 0.15, "medium": 0.3, "long": 0.5}
    ratio = ratio_map.get(length, 0.3)

    selected_indices = _select_sentence_indices(sentences, ratio)
    if not selected_indices:
        selected_indices = list(
            range(min(len(sentences), max(1, int(len(sentences) * ratio))))
        )

    summary_sentences = [sentences[idx] for idx in selected_indices]
    summary = " ".join(summary_sentences).strip()

    if summary and summary[-1] not in ".!?":
        summary += "."

    return clean_text_for_web(summary)
