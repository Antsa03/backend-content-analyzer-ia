import re
from typing import Dict, List, Sequence, Tuple

from .nlp_utils import (
    extract_keywords,
    filter_similar_terms,
    find_named_entities,
    restore_original_token,
    slice_context,
    split_sentences,
)


def extract_key_information(
    text: str,
    max_candidates: int = 150,
) -> Tuple[List[Dict[str, object]], List[str]]:
    """Analyse the text to extract high value sentences and keywords."""
    sentences = split_sentences(text, min_words=5)
    if not sentences:
        return [], []

    keyword_tuples = extract_keywords(sentences, max_keywords=max_candidates)
    if not keyword_tuples:
        return [], []

    keyword_scores = {kw: score for kw, score in keyword_tuples}
    keywords_lower = [kw.lower() for kw in keyword_scores.keys()]
    entities = find_named_entities(text)

    keyword_pool: List[str] = []
    for kw in keyword_scores.keys():
        keyword_pool.append(kw)
    keyword_pool.extend(entities)
    keyword_pool = list(dict.fromkeys(keyword_pool))

    info_items: List[Dict[str, object]] = []
    total_sentences = len(sentences)

    for idx, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        matched_keywords = [kw for kw in keywords_lower if kw in sentence_lower]
        if not matched_keywords:
            continue

        for kw_lower in matched_keywords:
            original_kw = restore_original_token(sentence, kw_lower)
            before, after = slice_context(sentence, kw_lower)

            info_items.append(
                {
                    "sentence": sentence,
                    "keyword": original_kw,
                    "keyword_lower": kw_lower,
                    "context_before": before,
                    "context_after": after,
                    "sentence_length": len(sentence.split()),
                    "position": idx / max(total_sentences - 1, 1),
                    "score": keyword_scores.get(kw_lower, 1.0),
                    "is_entity": original_kw in entities
                    or original_kw.upper() in entities,
                }
            )

    unique_items: Dict[Tuple[str, str], Dict[str, object]] = {}
    for item in info_items:
        key = (item["sentence"], item["keyword_lower"])
        if key not in unique_items or item["score"] > unique_items[key]["score"]:
            unique_items[key] = item

    ranked_items = sorted(
        unique_items.values(),
        key=lambda it: (
            it["score"],
            1.0 - it["position"],
            it["sentence_length"],
        ),
        reverse=True,
    )

    return ranked_items, keyword_pool


def generate_distractors(
    correct_answer: str,
    keyword_pool: Sequence[str],
    info: Dict[str, object],
    max_distractors: int = 6,
) -> List[str]:
    """Generate plausible distractors leveraging the keyword pool."""
    if not correct_answer:
        return ["Option A", "Option B", "Option C"]

    base_candidates = [term for term in keyword_pool if term and len(term) >= 3]

    context_before = info.get("context_before", "") if info else ""
    context_after = info.get("context_after", "") if info else ""
    context_terms = _extract_context_terms(context_before, context_after)
    base_candidates.extend(context_terms)

    ranked = filter_similar_terms(
        correct_answer, base_candidates, limit=max_distractors
    )

    distractors: List[str] = []
    for candidate in ranked:
        if candidate.lower() == correct_answer.lower():
            continue
        if candidate not in distractors:
            distractors.append(_match_case(candidate, correct_answer))
        if len(distractors) >= max_distractors:
            break

    if len(distractors) < 3:
        fallback_terms = [
            "Non précisé",
            "Indéterminé",
            "Non mentionné",
            "Information absente",
        ]
        for fallback in fallback_terms:
            if fallback.lower() != correct_answer.lower():
                distractors.append(fallback)
            if len(distractors) >= 3:
                break

    return distractors[:max_distractors]


def score_question_quality(
    question: str,
    options: Sequence[str],
    correct_answer: str,
) -> float:
    """Score question quality based on diversity and clarity."""
    if not question or not options or correct_answer not in options:
        return 0.0

    unique_options = len(set(options)) == len(options)
    length_score = min(len(question.split()) / 14.0, 1.0)
    diversity_score = _lexical_diversity(correct_answer, options)

    score = (
        0.4 * length_score + 0.45 * diversity_score + (0.15 if unique_options else 0.05)
    )
    return round(min(score, 1.0), 2)


def _lexical_diversity(correct: str, options: Sequence[str]) -> float:
    base = correct.lower()
    distances: List[float] = []
    for option in options:
        if option.lower() == base:
            continue
        dist = _levenshtein(base, option.lower())
        distances.append(dist / max(len(base), 1))

    if not distances:
        return 0.4

    avg_distance = sum(distances) / len(distances)
    return max(0.2, min(avg_distance, 1.0))


def _levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            insertions = prev[j] + 1
            deletions = curr[j - 1] + 1
            substitutions = prev[j - 1] + (ca != cb)
            curr.append(min(insertions, deletions, substitutions))
        prev = curr
    return prev[-1]


def _match_case(candidate: str, reference: str) -> str:
    if not reference:
        return candidate
    if reference.isupper():
        return candidate.upper()
    if reference[0].isupper():
        return candidate.capitalize()
    return candidate.lower()


def _extract_context_terms(*contexts: str) -> List[str]:
    terms: List[str] = []
    for context in contexts:
        if not context:
            continue
        for token in re.findall(
            r"\b[a-zàâäéèêëïîôùûüÿçœ'-]{3,}\b", context, re.IGNORECASE
        ):
            if len(token) >= 3:
                terms.append(token)
    return terms
