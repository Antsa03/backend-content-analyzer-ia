"""Utility helpers for lightweight NLP operations used by summarizer and quiz generator."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Simple French stop word list (extended with common liaison words)
FRENCH_STOP_WORDS = {
    "le",
    "la",
    "les",
    "un",
    "une",
    "des",
    "de",
    "du",
    "d'",
    "et",
    "ou",
    "mais",
    "dans",
    "pour",
    "par",
    "avec",
    "sans",
    "sous",
    "sur",
    "au",
    "aux",
    "ce",
    "cet",
    "cette",
    "ces",
    "mon",
    "ton",
    "son",
    "notre",
    "votre",
    "leur",
    "nos",
    "vos",
    "leurs",
    "qui",
    "que",
    "quoi",
    "dont",
    "où",
    "est",
    "sont",
    "était",
    "étaient",
    "été",
    "être",
    "avoir",
    "fait",
    "faire",
    "comme",
    "plus",
    "moins",
    "afin",
    "ainsi",
    "alors",
    "ceci",
    "cela",
    "cette",
    "celui",
    "celle",
    "leurs",
    "leurs",
    "nos",
    "vos",
    "très",
    "tout",
    "tous",
    "toutes",
    "aux",
    "lors",
    "depuis",
    "chez",
    "car",
    "donc",
    "or",
    "ni",
    "si",
    "quand",
    "comme",
}

_SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")
_WORD_REGEX = re.compile(r"\b[a-zàâäéèêëïîôùûüÿçœ'-]{3,}\b", re.IGNORECASE)


def split_sentences(text: str, min_words: int = 3) -> List[str]:
    """Split a text into sentences while keeping only informative ones."""
    if not text:
        return []
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in raw_sentences if len(s.split()) >= min_words]
    return sentences


def build_tfidf_matrix(sentences: Sequence[str]) -> Tuple[TfidfVectorizer, np.ndarray]:
    """Build a TF-IDF matrix for the provided sentences."""
    if not sentences:
        raise ValueError("TF-IDF requires at least one sentence")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=list(FRENCH_STOP_WORDS),
        ngram_range=(1, 2),
        min_df=1,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(sentences)
    return vectorizer, matrix


def rank_sentences(sentences: Sequence[str], matrix: np.ndarray) -> Dict[int, float]:
    """Rank sentences using a TextRank-like centrality derived from cosine similarity."""
    if not sentences:
        return {}

    if matrix.shape[0] == 1:
        return {0: 1.0}

    similarity = cosine_similarity(matrix)
    np.fill_diagonal(similarity, 0.0)

    scores = similarity.sum(axis=1)

    # Positional boost: early sentences often carry key context in structured documents.
    total_sentences = len(sentences)
    positional_weights = np.linspace(1.0, 0.7, total_sentences)
    boosted_scores = scores * positional_weights

    return {idx: float(score) for idx, score in enumerate(boosted_scores)}


def extract_keywords(
    sentences: Sequence[str],
    max_keywords: int = 40,
    min_score: float = 0.0,
) -> List[Tuple[str, float]]:
    """Extract significant keywords with their TF-IDF scores."""
    if not sentences:
        return []

    vectorizer, matrix = build_tfidf_matrix(sentences)
    feature_names = vectorizer.get_feature_names_out()
    keyword_scores = np.asarray(matrix.sum(axis=0)).ravel()

    scored_keywords = [
        (feature_names[i], float(keyword_scores[i]))
        for i in range(len(feature_names))
        if keyword_scores[i] > min_score and len(feature_names[i]) >= 3
    ]

    scored_keywords.sort(key=lambda item: item[1], reverse=True)
    return scored_keywords[:max_keywords]


def find_named_entities(text: str) -> List[str]:
    """Simple rule-based named entity extraction (capitalized words, acronyms)."""
    if not text:
        return []

    capitalized = re.findall(r"\b[A-ZÀÂÄÉÈÊËÏÎÔÙÛÜŸÇ][a-zàâäéèêëïîôùûüÿçœ]{2,}\b", text)
    acronyms = re.findall(r"\b[A-Z]{2,}\b", text)
    return list(dict.fromkeys(capitalized + acronyms))


def slice_context(sentence: str, keyword: str, window: int = 4) -> Tuple[str, str]:
    """Return context words around the first occurrence of keyword within a sentence."""
    tokens = sentence.split()
    keyword_lower = keyword.lower()
    index = -1
    span = 1
    for idx, token in enumerate(tokens):
        if keyword_lower in token.lower():
            index = idx
            break

    if index == -1 and " " in keyword_lower:
        phrase_tokens = [part for part in keyword_lower.split() if part]
        span = len(phrase_tokens)
        for idx in range(0, len(tokens) - span + 1):
            window_tokens = [
                t.strip(".,;:!?").lower() for t in tokens[idx : idx + span]
            ]
            if window_tokens == phrase_tokens:
                index = idx
                break

    if index == -1:
        return "", ""

    before = " ".join(tokens[max(0, index - window) : index]).strip()
    after = " ".join(tokens[index + span : index + span + window]).strip()
    return before, after


def restore_original_token(sentence: str, keyword: str) -> str:
    """Return the keyword with original casing from the sentence when possible."""
    pattern = re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
    match = pattern.search(sentence)
    if match:
        return match.group(0)
    return keyword


def filter_similar_terms(
    base: str, candidates: Iterable[str], limit: int = 10
) -> List[str]:
    """Filter candidate distractors to keep those close in length and characters."""
    base_lower = base.lower()
    filtered: List[Tuple[str, int]] = []

    for term in candidates:
        if not term or term.lower() == base_lower:
            continue
        distance = _levenshtein_distance(base_lower, term.lower())
        length_diff = abs(len(base_lower) - len(term))
        score = max(len(base_lower), len(term)) - distance - length_diff
        filtered.append((term, score))

    filtered.sort(key=lambda item: item[1], reverse=True)
    ranked = [term for term, _ in filtered]
    return ranked[:limit]


def _levenshtein_distance(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current_row = [i]
        for j, cb in enumerate(b, start=1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (ca != cb)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]
