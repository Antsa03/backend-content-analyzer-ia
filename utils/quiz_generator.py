import random
import re
from typing import Dict, List

from .text_cleaner import clean_text_for_web
from .header_detector import smart_clean_document
from .question_utils import (
    extract_key_information,
    generate_distractors,
    score_question_quality,
)


def generate_quiz(text: str, num_questions: int = 20):
    text = clean_text_for_web(text)
    text = smart_clean_document(text)

    info_items, keyword_pool = extract_key_information(text)
    if not info_items:
        return []

    questions: List[Dict[str, object]] = []
    used_sentences = set()
    keyword_usage: Dict[str, int] = {}

    for info in info_items:
        if len(questions) >= num_questions:
            break

        sentence = info["sentence"]
        keyword = info["keyword"]
        keyword_lower = info["keyword_lower"]

        if sentence in used_sentences:
            continue

        if keyword_usage.get(keyword_lower, 0) >= 2:
            continue

        question_prompt = _build_question_prompt(info)
        if not question_prompt:
            continue

        distractors = generate_distractors(keyword, keyword_pool, info)
        options = _assemble_options(keyword, distractors)
        if len(options) < 4:
            continue

        quality = score_question_quality(question_prompt, options, keyword)
        if quality < 0.55 and len(questions) >= max(1, int(num_questions * 0.5)):
            continue

        explanation = _build_explanation(info, keyword)

        cleaned_options = [clean_text_for_web(opt) for opt in options]
        correct_value = clean_text_for_web(keyword)
        if correct_value not in cleaned_options:
            cleaned_options.append(correct_value)
        random.shuffle(cleaned_options)
        correct_index = cleaned_options.index(correct_value)

        questions.append(
            {
                "question": clean_text_for_web(question_prompt),
                "options": cleaned_options,
                "correct_answer": correct_index,
                "explanation": clean_text_for_web(explanation),
            }
        )

        used_sentences.add(sentence)
        keyword_usage[keyword_lower] = keyword_usage.get(keyword_lower, 0) + 1

    return questions[:num_questions]


def _build_question_prompt(info: Dict[str, object]) -> str:
    sentence = info["sentence"]
    keyword = info["keyword"]
    context_before = info.get("context_before", "")
    context_after = info.get("context_after", "")

    blanked = _replace_first_occurrence(sentence, keyword, "______")

    if context_before and context_after:
        return f"Dans le passage : « ...{context_before} ______ {context_after}... », quel terme complète la phrase ?"

    if info.get("is_entity"):
        return f"Quel élément est mentionné dans cette phrase : « {blanked} » ?"

    return f"Complétez la phrase suivante : {blanked}"


def _build_explanation(info: Dict[str, object], correct_answer: str) -> str:
    sentence = info["sentence"]
    details = [f"La réponse correcte est « {correct_answer} »."]
    if info.get("is_entity"):
        details.append("Ce terme est identifié comme une entité importante du texte.")
    details.append(
        f"Extrait : « {sentence[:140]}{'...' if len(sentence) > 140 else ''} »"
    )
    return " ".join(details)


def _replace_first_occurrence(sentence: str, keyword: str, placeholder: str) -> str:
    pattern = re.compile(rf"\b{re.escape(keyword)}\b")
    return pattern.sub(placeholder, sentence, count=1)


def _assemble_options(correct_answer: str, distractors: List[str]) -> List[str]:
    options = []
    seen = set()
    for candidate in distractors:
        normalized = candidate.strip()
        if not normalized:
            continue
        if normalized.lower() == correct_answer.lower():
            continue
        if normalized.lower() in seen:
            continue
        options.append(normalized)
        seen.add(normalized.lower())
        if len(options) >= 3:
            break

    cleaned_correct = clean_text_for_web(correct_answer)
    if cleaned_correct not in options:
        options.append(cleaned_correct)

    fallback_pool = [
        "Réponse non mentionnée",
        "Information absente",
        "Aucune des réponses",
        "Option inconnue",
    ]

    for filler in fallback_pool:
        if len(options) >= 4:
            break
        if filler.lower() in seen or filler.lower() == correct_answer.lower():
            continue
        options.append(filler)
        seen.add(filler.lower())

    return options
