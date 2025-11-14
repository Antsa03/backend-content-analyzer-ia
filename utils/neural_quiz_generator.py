"""
Générateur de quiz amélioré utilisant les réseaux de neurones (Deep Learning).
Combine l'approche extractive avec des modèles Transformer pour la génération de questions.
"""

import random
from typing import Dict, List, Optional
import logging

from .text_cleaner import clean_text_for_web
from .header_detector import smart_clean_document
from .nlp_utils import split_sentences, extract_keywords, find_named_entities

logger = logging.getLogger(__name__)

# Flag pour activer/désactiver le mode Deep Learning
USE_NEURAL_MODEL = True

try:
    from models.neural_models import ModelManager

    model_manager = ModelManager()
    logger.info("✅ Modèles neuronaux de quiz disponibles")
except ImportError as e:
    USE_NEURAL_MODEL = False
    logger.warning(f"⚠️  Modèles neuronaux non disponibles: {e}")
    logger.info("📊 Utilisation du mode extractif classique")


def generate_quiz_neural(text: str, num_questions: int = 20) -> List[Dict]:
    """
    Génère un quiz en français en utilisant un réseau de neurones Transformer (T5/FLAN-T5).

    Cette fonction utilise un modèle pré-entraîné optimisé pour le français
    avec des prompts spécialement conçus pour générer des questions naturelles
    et des distracteurs plausibles.

    Args:
        text: Texte source en français
        num_questions: Nombre de questions à générer (défaut: 20)

    Returns:
        Liste de questions avec options et réponses validées pour le français
    """
    text = clean_text_for_web(text)
    text = smart_clean_document(text)

    if not USE_NEURAL_MODEL:
        # Fallback vers l'approche classique
        from .quiz_generator import generate_quiz

        return generate_quiz(text, num_questions)

    try:
        # Extraction des phrases et entités importantes
        sentences = split_sentences(text, min_words=8)
        if not sentences:
            return []

        # Extraction des mots-clés avec TF-IDF
        keyword_tuples = extract_keywords(sentences, max_keywords=50)
        entities = find_named_entities(text)

        # Récupération du modèle neuronal
        quiz_gen = model_manager.get_quiz_generator()

        questions = []
        used_sentences = set()

        # Sélection des phrases les plus informatives
        selected_sentences = _select_informative_sentences(
            sentences, keyword_tuples, min(num_questions * 2, len(sentences))
        )

        for sentence in selected_sentences:
            if len(questions) >= num_questions:
                break

            if sentence in used_sentences:
                continue

            # Extraction de la réponse candidate (entité ou mot-clé)
            answer = _extract_answer_from_sentence(sentence, keyword_tuples, entities)
            if not answer:
                continue

            # Génération de la question avec le réseau de neurones et prompts FR optimisés
            question_text = quiz_gen.generate_question(
                context=sentence,
                answer=answer,
                max_length=80,  # ✅ Augmenté pour questions FR complètes
            )

            # Validation plus stricte pour questions françaises
            if (
                not question_text or len(question_text.split()) < 5
            ):  # ✅ Min 5 mots au lieu de 4
                continue

            # ✅ Vérifier que c'est une vraie question en français
            if not _is_valid_french_question(question_text):
                logger.debug(f"❌ Question invalide ignorée: {question_text}")
                continue

            # Génération des distracteurs (mauvaises réponses)
            distractors = _generate_neural_distractors(
                quiz_gen, question_text, answer, sentence, entities, keyword_tuples
            )

            # Assemblage des options
            options = distractors[:3]
            options.append(answer)
            random.shuffle(options)

            correct_index = options.index(answer)

            questions.append(
                {
                    "question": clean_text_for_web(question_text),
                    "options": [clean_text_for_web(opt) for opt in options],
                    "correct_answer": correct_index,
                    "explanation": f"La réponse correcte est '{answer}'. Extrait : {sentence[:120]}...",
                }
            )

            used_sentences.add(sentence)

        logger.info(f"✅ {len(questions)} questions générées avec le modèle neuronal")
        return questions[:num_questions]

    except Exception as e:
        logger.error(f"❌ Erreur avec le modèle neuronal de quiz: {e}")
        logger.info("🔄 Basculement vers le générateur classique")
        from .quiz_generator import generate_quiz

        return generate_quiz(text, num_questions)


def _select_informative_sentences(
    sentences: List[str], keyword_tuples: List[tuple], top_n: int
) -> List[str]:
    """Sélectionne les phrases les plus informatives basées sur les mots-clés."""
    keyword_scores = {kw: score for kw, score in keyword_tuples}

    sentence_scores = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = sum(
            keyword_scores.get(kw, 0)
            for kw in keyword_scores.keys()
            if kw.lower() in sentence_lower
        )
        sentence_scores.append((sentence, score))

    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    return [sent for sent, _ in sentence_scores[:top_n]]


def _extract_answer_from_sentence(
    sentence: str, keyword_tuples: List[tuple], entities: List[str]
) -> Optional[str]:
    """Extrait une réponse candidate (entité nommée ou mot-clé important)."""
    sentence_lower = sentence.lower()

    # Priorité aux entités nommées
    for entity in entities:
        if entity.lower() in sentence_lower:
            return entity

    # Sinon, utilise les mots-clés TF-IDF
    for keyword, score in keyword_tuples:
        if keyword.lower() in sentence_lower and len(keyword) >= 3:
            return keyword

    return None


def _generate_neural_distractors(
    quiz_gen,
    question: str,
    correct_answer: str,
    context: str,
    entities: List[str],
    keyword_tuples: List[tuple],
    max_attempts: int = 5,
) -> List[str]:
    """
    Génère des distracteurs (mauvaises réponses) en utilisant le réseau de neurones.
    Combine génération neuronale et extraction d'entités/mots-clés.
    """
    distractors = []

    # 1. Génération neuronale de distracteurs
    for _ in range(min(2, max_attempts)):
        try:
            neural_distractor = quiz_gen.generate_distractor(
                question=question, correct_answer=correct_answer, context=context
            )
            if (
                neural_distractor
                and neural_distractor.lower() != correct_answer.lower()
                and neural_distractor not in distractors
            ):
                distractors.append(neural_distractor)
        except Exception as e:
            logger.debug(f"Erreur génération distracteur: {e}")

    # 2. Compléter avec des entités/mots-clés du texte
    candidates = list(set(entities + [kw for kw, _ in keyword_tuples[:20]]))
    random.shuffle(candidates)

    for candidate in candidates:
        if len(distractors) >= 3:
            break
        if (
            candidate.lower() != correct_answer.lower()
            and candidate not in distractors
            and len(candidate) >= 3
        ):
            distractors.append(candidate)

    # 3. Fallback si pas assez de distracteurs
    fallback_distractors = [
        "Aucune de ces réponses",
        "Information non mentionnée",
        "Donnée absente du texte",
        "Réponse indéterminée",
    ]

    for fallback in fallback_distractors:
        if len(distractors) >= 3:
            break
        if fallback.lower() != correct_answer.lower():
            distractors.append(fallback)

    return distractors[:3]


def _is_valid_french_question(question: str) -> bool:
    """
    Vérifie si une question générée est valide en français.

    Args:
        question: Question à valider

    Returns:
        True si la question est valide, False sinon
    """
    if not question or len(question.strip()) < 5:
        return False

    question_lower = question.lower().strip()

    # Doit se terminer par un point d'interrogation
    if not question.endswith("?"):
        return False

    # Mots de début de question valides en français
    valid_starters = [
        "quel",
        "quelle",
        "quels",
        "quelles",
        "qui",
        "que",
        "quoi",
        "où",
        "quand",
        "comment",
        "pourquoi",
        "combien",
        "lequel",
        "laquelle",
        "lesquels",
        "lesquelles",
        "est-ce",
        "peut-on",
        "doit-on",
        "faut-il",
        "dans",
        "selon",
        "à partir",
        "complétez",
        "citez",
        "nommez",
        "identifiez",
        "définissez",
        "expliquez",
    ]

    starts_valid = any(question_lower.startswith(starter) for starter in valid_starters)

    # Vérifier absence de mots interdits
    forbidden_words = ["[unk]", "<unk>", "undefined", "null", "error"]
    has_forbidden = any(word in question_lower for word in forbidden_words)

    return starts_valid and not has_forbidden


def generate_quiz_hybrid(text: str, num_questions: int = 20) -> dict:
    """
    Génère deux quiz : un avec Deep Learning et un avec l'approche classique.
    Permet de comparer les deux approches.

    Returns:
        Dict avec 'neural_quiz' et 'classic_quiz'
    """
    from .quiz_generator import generate_quiz as generate_classic

    neural_quiz = generate_quiz_neural(text, num_questions)
    classic_quiz = generate_classic(text, num_questions)

    return {
        "neural_quiz": neural_quiz,
        "classic_quiz": classic_quiz,
        "method_used": "hybrid",
    }
