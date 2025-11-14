"""
Modèles de Deep Learning basés sur les réseaux de neurones Transformer
pour la génération de résumés et de quiz.
"""

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        T5ForConditionalGeneration,
        pipeline,
    )

    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    TORCH_AVAILABLE = False
    print(f"⚠️  PyTorch non disponible: {e}")
    print("ℹ️  Le système utilisera uniquement le mode TF-IDF classique.")
    print("ℹ️  Pour activer le Deep Learning, installez Visual C++ Redistributable:")
    print("ℹ️  https://aka.ms/vs/17/release/vc_redist.x64.exe")

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class NeuralSummarizer:
    """
    Résumeur basé sur un réseau de neurones Transformer (architecture encoder-decoder).
    Utilise des modèles pré-entraînés comme mBART ou T5 pour générer des résumés abstractifs.
    """

    def __init__(self, model_name: str = "moussaKam/barthez-orangesum-abstract"):
        """
        Initialise le modèle de résumé neuronal.

        Args:
            model_name: Nom du modèle Hugging Face à utiliser
                       - "moussaKam/barthez-orangesum-abstract" : BART français (RECOMMANDÉ)
                       - "plguillou/t5-base-fr-sum-cnndm" : T5 français pour résumés
                       - "facebook/mbart-large-50" : multilingue (français inclus)
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch n'est pas disponible. Installez Visual C++ Redistributable."
            )

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 Chargement du modèle de résumé neuronal: {model_name}")
        logger.info(f"🖥️  Device utilisé: {self.device}")

        try:
            # Chargement avec use_fast=False pour éviter les erreurs de tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=False,  # ✅ Force le tokenizer lent (compatible SentencePiece)
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()  # Mode évaluation (pas d'entraînement)
            logger.info("✅ Modèle de résumé chargé avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
            raise

    def generate_summary(
        self,
        text: str,
        max_length: int = 250,
        min_length: int = 80,
        num_beams: int = 6,
        repetition_penalty: float = 1.2,
    ) -> str:
        """
        Génère un résumé abstractif en utilisant le réseau de neurones.

        Args:
            text: Texte source à résumer
            max_length: Longueur maximale du résumé généré
            min_length: Longueur minimale du résumé généré
            num_beams: Nombre de beams pour la recherche (beam search)
            repetition_penalty: Pénalité pour les répétitions (>1 = moins de répétitions)

        Returns:
            Résumé généré par le modèle
        """
        if not text or len(text.strip()) < 50:
            return text

        try:
            # Nettoyage et préparation du texte
            text = text.strip()

            # Tokenisation du texte d'entrée avec troncature intelligente
            inputs = self.tokenizer(
                text,
                max_length=512,  # ✅ Réduit à 512 pour éviter les hallucinations
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Génération du résumé avec paramètres optimisés pour BARThez
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=4,  # ✅ Augmenté de 3 à 4 pour éviter plus de répétitions
                    length_penalty=1.2,  # ✅ Réduit pour favoriser résumés plus courts et cohérents
                    do_sample=False,  # Déterministe (pas de temperature)
                    forced_eos_token_id=self.tokenizer.eos_token_id,  # ✅ Force la fin propre
                )

            # Décodage du résultat
            summary = self.tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # ✅ Validation et nettoyage du résumé
            summary = self._clean_and_validate_summary(summary.strip(), text)

            return summary

        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération du résumé: {e}")
            # Fallback : retourne les N premiers caractères
            return text[: max_length * 5]

    def _clean_and_validate_summary(self, summary: str, original_text: str) -> str:
        """
        Nettoie et valide le résumé généré pour détecter les hallucinations.

        Args:
            summary: Résumé généré par le modèle
            original_text: Texte source original

        Returns:
            Résumé nettoyé et validé
        """
        if not summary or len(summary.strip()) < 10:
            return original_text[:500]

        # Détecter les répétitions excessives de mots
        words = summary.split()

        # Si le résumé est trop long par rapport à l'original, c'est suspect
        if len(words) > len(original_text.split()) * 0.8:
            logger.warning("⚠️  Résumé trop long, troncature")
            summary = " ".join(words[: len(original_text.split()) // 2])

        # Détecter les répétitions de patterns (ex: "K K K K")
        from collections import Counter

        word_counts = Counter(words)

        # Si un mot (hors mots courants) apparaît plus de 5 fois, c'est suspect
        common_words = {
            "le",
            "la",
            "les",
            "de",
            "des",
            "un",
            "une",
            "et",
            "ou",
            "à",
            "du",
            "en",
        }
        suspicious = False
        for word, count in word_counts.items():
            if word.lower() not in common_words and count > 5:
                logger.warning(f"⚠️  Répétition excessive détectée: '{word}' x{count}")
                suspicious = True
                break

        # Si répétitions suspectes, tronquer au premier problème
        if suspicious:
            cleaned_sentences = []
            sentences = summary.split(".")

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Vérifier si la phrase contient des répétitions anormales
                sentence_words = sentence.split()
                if len(sentence_words) < 3:
                    continue

                # Vérifier les trigrammes répétés
                has_repetition = False
                for i in range(len(sentence_words) - 2):
                    trigram = " ".join(sentence_words[i : i + 3])
                    if sentence.count(trigram) > 1:
                        has_repetition = True
                        break

                if not has_repetition:
                    cleaned_sentences.append(sentence)
                else:
                    logger.warning(
                        f"⚠️  Phrase avec répétitions ignorée: {sentence[:50]}..."
                    )
                    break  # Arrêter dès qu'on trouve une phrase suspecte

            if cleaned_sentences:
                summary = ". ".join(cleaned_sentences) + "."
            else:
                # Si tout est suspect, fallback sur le début du texte original
                logger.error(
                    "❌ Résumé complètement incohérent, utilisation du texte source"
                )
                return original_text[:500] + "..."

        return summary.strip()


class NeuralQuizGenerator:
    """
    Générateur de quiz basé sur un réseau de neurones Transformer (T5).
    Génère des questions et réponses à partir du texte source.
    """

    def __init__(self, model_name: str = "google/flan-t5-base"):
        """
        Initialise le modèle de génération de questions.

        Args:
            model_name: Nom du modèle Hugging Face à utiliser
                       - "google/flan-t5-base" : T5 multilingue optimisé
                       - "mrm8488/t5-base-finetuned-question-generation-ap" : T5 pour QG
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch n'est pas disponible. Installez Visual C++ Redistributable."
            )

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 Chargement du modèle de génération de quiz: {model_name}")
        logger.info(f"🖥️  Device utilisé: {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("✅ Modèle de quiz chargé avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
            raise

    def generate_question(
        self,
        context: str,
        answer: Optional[str] = None,
        max_length: int = 80,
    ) -> str:
        """
        Génère une question en français à partir d'un contexte et optionnellement d'une réponse.

        Args:
            context: Texte contexte en français
            answer: Réponse cible (optionnel)
            max_length: Longueur max de la question

        Returns:
            Question générée en français
        """
        try:
            # Prompt engineering optimisé pour le français
            if answer:
                # Prompt français structuré pour T5
                input_text = (
                    f"Générez une question en français basée sur cette information. "
                    f"Réponse attendue: {answer}. Contexte: {context}"
                )
            else:
                # Prompt alternatif sans réponse
                input_text = f"Posez une question pertinente en français à partir du texte suivant: {context}"

            inputs = self.tokenizer(
                input_text,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                question_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    min_length=8,  # ✅ Minimum pour cohérence
                    num_beams=6,  # ✅ Augmenté pour meilleure qualité
                    repetition_penalty=1.3,  # ✅ Évite répétitions
                    no_repeat_ngram_size=3,  # ✅ Bloque trigrammes répétés
                    length_penalty=1.0,  # ✅ Neutre
                    do_sample=False,  # Déterministe
                )

            question = self.tokenizer.decode(
                question_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # ✅ Validation et nettoyage de la question
            question = self._validate_and_clean_question(
                question.strip(), context, answer
            )

            return question

        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération de question: {e}")
            # Fallback intelligent en français
            if answer:
                return f"Quel est le terme manquant : {context[:80]}... (Réponse: ______) ?"
            return f"Quelle information peut-on extraire de : {context[:60]}... ?"

    def generate_distractor(
        self,
        question: str,
        correct_answer: str,
        context: str,
    ) -> str:
        """
        Génère un distracteur (mauvaise réponse plausible) en français pour une question.

        Args:
            question: La question en français
            correct_answer: La bonne réponse
            context: Le contexte en français

        Returns:
            Un distracteur plausible en français
        """
        try:
            # Prompt français optimisé pour distracteurs
            input_text = (
                f"Générez une réponse incorrecte mais plausible en français. "
                f"Question: {question} "
                f"Bonne réponse: {correct_answer} "
                f"Contexte: {context[:200]}"
            )

            inputs = self.tokenizer(
                input_text,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                distractor_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=40,  # ✅ Augmenté pour distracteurs français
                    min_length=3,  # ✅ Minimum de cohérence
                    num_beams=5,  # ✅ Augmenté pour qualité
                    repetition_penalty=1.4,  # ✅ Forte pénalité
                    do_sample=True,  # Échantillonnage pour diversité
                    top_k=40,  # ✅ Réduit pour meilleure qualité
                    top_p=0.92,  # ✅ Ajusté pour cohérence
                    temperature=0.85,  # ✅ Créativité contrôlée
                )

            distractor = self.tokenizer.decode(
                distractor_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # ✅ Validation du distracteur
            distractor = self._validate_distractor(distractor.strip(), correct_answer)

            return distractor

        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération de distracteur: {e}")
            # Fallback intelligent en français
            fallbacks = [
                "Autre réponse possible",
                "Information non mentionnée",
                "Donnée absente du contexte",
            ]
            import random

            return random.choice(fallbacks)

    def _validate_and_clean_question(
        self, question: str, context: str, answer: Optional[str]
    ) -> str:
        """
        Valide et nettoie une question générée en français.

        Args:
            question: Question générée
            context: Contexte source
            answer: Réponse attendue

        Returns:
            Question validée et nettoyée
        """
        if not question or len(question.strip()) < 5:
            # Fallback si question vide
            if answer:
                return (
                    f"Quel est le terme manquant dans ce contexte : {context[:80]}... ?"
                )
            return f"Quelle information peut-on extraire du texte suivant ?"

        # Nettoyage de base
        question = question.strip()

        # Vérifier les mots interdits
        forbidden = ["[UNK]", "<unk>", "undefined", "null"]
        for word in forbidden:
            if word.lower() in question.lower():
                logger.warning(f"⚠️  Mot interdit détecté dans la question: {word}")
                if answer:
                    return f"Quelle est la réponse correcte concernant {answer} dans ce contexte ?"
                return f"Quelle information principale est présente dans ce passage ?"

        # S'assurer qu'il y a un point d'interrogation
        if not question.endswith("?"):
            # Vérifier si c'est bien une question en français
            question_starters = [
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
            ]
            starts_with_question = any(
                question.lower().startswith(starter) for starter in question_starters
            )
            if starts_with_question:
                question = question.rstrip(".") + "?"

        # Limiter la longueur (max 200 caractères)
        if len(question) > 200:
            question = question[:197] + "...?"

        # Capitaliser le début
        if question and question[0].islower():
            question = question[0].upper() + question[1:]

        return question

    def _validate_distractor(self, distractor: str, correct_answer: str) -> str:
        """
        Valide un distracteur généré.

        Args:
            distractor: Distracteur généré
            correct_answer: Réponse correcte (ne doit pas être identique)

        Returns:
            Distracteur validé
        """
        if not distractor or len(distractor.strip()) < 2:
            return "Réponse alternative"

        distractor = distractor.strip()

        # Vérifier qu'il n'est pas identique à la bonne réponse
        if distractor.lower() == correct_answer.lower():
            logger.warning(f"⚠️  Distracteur identique à la bonne réponse")
            return f"Non-{correct_answer}"

        # Nettoyer les caractères spéciaux inutiles
        distractor = distractor.strip(".,;:!?")

        # Limiter la longueur (max 100 caractères)
        if len(distractor) > 100:
            distractor = distractor[:97] + "..."

        return distractor


class ModelManager:
    """
    Gestionnaire centralisé des modèles neuronaux.
    Implémente le pattern Singleton pour éviter de charger plusieurs fois les modèles.
    """

    _instance = None
    _summarizer: Optional[NeuralSummarizer] = None
    _quiz_generator: Optional[NeuralQuizGenerator] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_summarizer(
        self, model_name: str = "moussaKam/barthez-orangesum-abstract"
    ) -> NeuralSummarizer:
        """Retourne l'instance du résumeur neuronal (singleton)."""
        if self._summarizer is None:
            self._summarizer = NeuralSummarizer(model_name)
        return self._summarizer

    def get_quiz_generator(
        self, model_name: str = "google/flan-t5-base"
    ) -> NeuralQuizGenerator:
        """Retourne l'instance du générateur de quiz neuronal (singleton)."""
        if self._quiz_generator is None:
            self._quiz_generator = NeuralQuizGenerator(model_name)
        return self._quiz_generator

    def unload_models(self):
        """Libère la mémoire en déchargeant les modèles."""
        if self._summarizer:
            del self._summarizer
            self._summarizer = None
        if self._quiz_generator:
            del self._quiz_generator
            self._quiz_generator = None
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("🗑️  Modèles déchargés de la mémoire")
