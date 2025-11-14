"""
Résumeur amélioré utilisant les réseaux de neurones (Deep Learning).
Combine l'approche TF-IDF classique avec des modèles Transformer.
"""

from typing import Optional
import logging

from .text_cleaner import clean_text_for_web
from .header_detector import smart_clean_document

logger = logging.getLogger(__name__)

# Flag pour activer/désactiver le mode Deep Learning
USE_NEURAL_MODEL = True

try:
    from models.neural_models import ModelManager

    model_manager = ModelManager()
    logger.info("✅ Modèles neuronaux disponibles")
except ImportError as e:
    USE_NEURAL_MODEL = False
    logger.warning(f"⚠️  Modèles neuronaux non disponibles: {e}")
    logger.info("📊 Utilisation du mode TF-IDF classique")


def generate_summary_neural(text: str, length: str = "medium") -> str:
    """
    Génère un résumé en utilisant un réseau de neurones Transformer.

    La longueur du résumé est calculée dynamiquement en fonction du texte source :
    - short : réduction à 80% (conserve 20% du texte)
    - medium : réduction à 60% (conserve 40% du texte)
    - long : réduction à 45% (conserve 55% du texte)

    Args:
        text: Texte source à résumer
        length: "short", "medium" ou "long"

    Returns:
        Résumé généré par le réseau de neurones
    """
    text = clean_text_for_web(text)
    text = smart_clean_document(text)

    if not USE_NEURAL_MODEL:
        # Fallback vers l'approche classique TF-IDF
        from .summarizer import generate_summary

        return generate_summary(text, length)

    try:
        # Calcul de la longueur du texte source en mots
        word_count = len(text.split())

        # Configuration basée sur les pourcentages de réduction
        # short: garde 20% (réduit 80%)
        # medium: garde 40% (réduit 60%)
        # long: garde 55% (réduit 45%)
        reduction_config = {
            "short": {
                "reduction_percent": 0.80,  # Réduction à 80% (garde 20%)
                "keep_percent": 0.20,
                "num_beams": 5,
                "repetition_penalty": 1.5,  # ✅ Augmenté de 1.2 à 1.5
            },
            "medium": {
                "reduction_percent": 0.60,  # Réduction à 60% (garde 40%)
                "keep_percent": 0.40,
                "num_beams": 6,
                "repetition_penalty": 1.5,  # ✅ Augmenté de 1.2 à 1.5
            },
            "long": {
                "reduction_percent": 0.45,  # Réduction à 45% (garde 55%)
                "keep_percent": 0.55,
                "num_beams": 6,
                "repetition_penalty": 1.4,  # ✅ Augmenté de 1.15 à 1.4
            },
        }

        config = reduction_config.get(length, reduction_config["medium"])

        # Calcul dynamique des longueurs basé sur le pourcentage
        target_length = int(word_count * config["keep_percent"])

        # Définir min et max avec une marge de flexibilité
        min_length = max(30, int(target_length * 0.7))  # 70% de la cible minimum
        max_length = min(
            400, int(target_length * 1.2)
        )  # ✅ Réduit à 400 max et marge à 20%

        # Sécurité : limites absolues plus strictes
        min_length = max(min_length, 30)  # Au moins 30 mots
        max_length = min(max_length, 400)  # ✅ Au plus 400 mots (réduit de 512)

        # S'assurer que min < max
        if min_length >= max_length:
            max_length = min_length + 50

        logger.info(
            f"📊 Texte source: {word_count} mots | "
            f"Réduction: {int(config['reduction_percent']*100)}% | "
            f"Cible: {target_length} mots | "
            f"Plage: {min_length}-{max_length} mots"
        )

        # Récupération du modèle neuronal (BARThez optimisé pour le français)
        summarizer = model_manager.get_summarizer(
            model_name="moussaKam/barthez-orangesum-abstract"
        )

        # Génération du résumé avec les paramètres calculés dynamiquement
        summary = summarizer.generate_summary(
            text,
            max_length=max_length,
            min_length=min_length,
            num_beams=config["num_beams"],
            repetition_penalty=config["repetition_penalty"],
        )

        return clean_text_for_web(summary)

    except Exception as e:
        logger.error(f"❌ Erreur avec le modèle neuronal: {e}")
        logger.info("🔄 Basculement vers TF-IDF classique")
        from .summarizer import generate_summary

        return generate_summary(text, length)


def generate_summary_hybrid(text: str, length: str = "medium") -> dict:
    """
    Génère deux résumés : un avec Deep Learning et un avec TF-IDF.
    Permet de comparer les deux approches.

    Returns:
        Dict avec 'neural_summary' et 'tfidf_summary'
    """
    from .summarizer import generate_summary as generate_tfidf

    neural_summary = generate_summary_neural(text, length)
    tfidf_summary = generate_tfidf(text, length)

    return {
        "neural_summary": neural_summary,
        "tfidf_summary": tfidf_summary,
        "method_used": "hybrid",
    }
