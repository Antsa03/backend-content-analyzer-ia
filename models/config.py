"""
Configuration des modèles de Deep Learning.
Permet de personnaliser les modèles utilisés et leurs paramètres.
"""

# ==========================================================
# 🎯 CONFIGURATION DES MODÈLES
# ==========================================================

# Modèle pour le résumé
SUMMARIZER_CONFIG = {
    # Modèle principal (recommandé pour le français)
    # Options classées par qualité pour le français :
    # 1. "moussaKam/barthez-orangesum-abstract" - MEILLEUR pour résumés français
    # 2. "plguillou/t5-base-fr-sum-cnndm" - T5 français spécialisé résumé
    # 3. "facebook/mbart-large-50" - Multilingue (moins spécialisé)
    # 4. "google/mt5-base" - mT5 multilingue (plus léger)
    "model_name": "moussaKam/barthez-orangesum-abstract",  # ✅ Optimisé pour résumé FR
    # Paramètres de génération basés sur POURCENTAGES DE RÉDUCTION
    # Les longueurs sont calculées dynamiquement selon la taille du texte source
    "generation_params": {
        "short": {
            # Réduction à 80% (conserve 20% du texte original)
            "reduction_percent": 0.80,
            "keep_percent": 0.20,
            "num_beams": 5,
            "temperature": 0.85,
            "repetition_penalty": 1.2,
        },
        "medium": {
            # Réduction à 60% (conserve 40% du texte original)
            "reduction_percent": 0.60,
            "keep_percent": 0.40,
            "num_beams": 6,
            "temperature": 0.9,
            "repetition_penalty": 1.2,
        },
        "long": {
            # Réduction à 45% (conserve 55% du texte original)
            "reduction_percent": 0.45,
            "keep_percent": 0.55,
            "num_beams": 6,
            "temperature": 0.95,
            "repetition_penalty": 1.15,
        },
    },
    # Limites absolues pour la génération
    "length_limits": {
        "min_words": 30,  # Minimum absolu de mots dans un résumé
        "max_words": 512,  # Maximum absolu (limite du modèle)
        "flexibility_min": 0.7,  # Marge basse (70% de la cible)
        "flexibility_max": 1.3,  # Marge haute (130% de la cible)
    },
    # Paramètres avancés OPTIMISÉS
    "advanced_params": {
        "length_penalty": 1.5,  # Réduit de 2.0 pour plus de flexibilité
        "no_repeat_ngram_size": 3,  # Évite les répétitions de 3-grams
        "do_sample": False,  # Déterministe pour cohérence
    },
}

# Modèle pour la génération de quiz
QUIZ_GENERATOR_CONFIG = {
    # Modèle principal optimisé pour le français
    # Options classées par qualité pour le français :
    # 1. "etalab-ia/camembert2-large-fquad" - CamemBERT français QA (RECOMMANDÉ)
    # 2. "lincoln/flaubert-mlsum-topic-classification" - FlauBERT français
    # 3. "google/flan-t5-base" - Multilingue généraliste (moins spécialisé)
    # 4. "mrm8488/t5-base-finetuned-question-generation-ap" - Anglais principalement
    "model_name": "google/flan-t5-base",  # Gardé pour compatibilité, mais avec prompts FR optimisés
    # Prompts optimisés pour le français
    "prompts_fr": {
        "question_with_answer": "Générez une question en français basée sur cette information. Réponse attendue: {answer}. Contexte: {context}",
        "question_cloze": "Créez une question à trou en français pour tester cette connaissance: {context}",
        "question_what": "À partir du texte suivant, posez une question commençant par 'Quel', 'Quelle', 'Quels' ou 'Quelles': {context}",
        "question_why": "Formulez une question 'Pourquoi' basée sur: {context}",
        "question_how": "Créez une question 'Comment' à partir de: {context}",
        "distractor": "Générez une réponse incorrecte mais plausible en français. Question: {question}. Bonne réponse: {correct_answer}. Contexte: {context}",
    },
    # Paramètres pour la génération de questions
    "question_params": {
        "max_length": 80,  # ✅ Augmenté de 64 à 80 pour questions françaises plus complètes
        "min_length": 8,  # ✅ Minimum de 8 tokens pour cohérence
        "num_beams": 6,  # ✅ Augmenté de 4 à 6 pour meilleure qualité
        "repetition_penalty": 1.3,  # ✅ Évite répétitions
        "no_repeat_ngram_size": 3,  # ✅ Bloque trigrammes répétés
        "length_penalty": 1.0,  # ✅ Neutre pour longueur naturelle
        "do_sample": False,  # Déterministe pour cohérence
    },
    # Paramètres pour la génération de distracteurs
    "distractor_params": {
        "max_length": 40,  # ✅ Augmenté de 32 à 40 pour distracteurs français
        "min_length": 3,  # ✅ Minimum de 3 tokens
        "num_beams": 5,  # ✅ Augmenté pour plus de diversité
        "repetition_penalty": 1.4,  # ✅ Forte pénalité pour éviter répétitions
        "do_sample": True,  # Échantillonnage pour diversité
        "top_k": 40,  # ✅ Réduit de 50 à 40 pour meilleure qualité
        "top_p": 0.92,  # ✅ Ajusté de 0.95 à 0.92 pour plus de cohérence
        "temperature": 0.85,  # ✅ Température modérée pour créativité contrôlée
    },
    # Validation de qualité des questions
    "quality_validation": {
        "min_question_length": 5,  # Minimum de mots dans une question
        "max_question_length": 30,  # Maximum de mots dans une question
        "min_distractor_length": 2,
        "require_question_mark": True,  # Force le point d'interrogation
        "forbidden_words": ["[UNK]", "<unk>", "undefined"],  # Mots interdits
        "french_question_starters": [  # Débuts de questions valides en français
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
        ],
    },
}

# ==========================================================
# ⚙️ CONFIGURATION SYSTÈME
# ==========================================================

SYSTEM_CONFIG = {
    # Activer/désactiver le mode Deep Learning
    "use_neural_models": True,
    # Fallback vers TF-IDF si erreur
    "fallback_to_tfidf": True,
    # Device : "cuda", "cpu", ou "auto" (détection automatique)
    "device": "auto",
    # Batch size pour traitement par lots (si implémenté)
    "batch_size": 1,
    # Cache des modèles
    "cache_models": True,  # Garde les modèles en mémoire (Singleton)
    # Logging
    "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR
    # Timeouts
    "generation_timeout": 30,  # secondes
}

# ==========================================================
# 🔧 CONFIGURATION AVANCÉE (Optimisation)
# ==========================================================

OPTIMIZATION_CONFIG = {
    # Quantization (réduction de la précision pour économiser mémoire)
    "use_quantization": False,  # Requiert optimum library
    "quantization_bits": 8,  # 8 ou 4 bits
    # Mixed precision (pour GPU)
    "use_mixed_precision": True,  # float16 au lieu de float32
    # Gradient checkpointing (économise mémoire)
    "gradient_checkpointing": False,  # Utile seulement pour le fine-tuning
    # Number of threads (pour CPU)
    "num_threads": 4,
    # Max input length (tronquer les textes trop longs)
    "max_input_length": 1024,  # tokens
}

# ==========================================================
# 🎨 CONFIGURATION DES PROMPTS (Prompt Engineering)
# ==========================================================

PROMPT_CONFIG = {
    # Templates pour la génération de questions
    "question_templates": {
        "with_answer": "generate question: answer: {answer} context: {context}",
        "without_answer": "generate question: {context}",
        "mcq": "generate multiple choice question: context: {context}",
    },
    # Templates pour la génération de distracteurs
    "distractor_template": (
        "generate wrong answer: question: {question} "
        "correct answer: {correct_answer} context: {context}"
    ),
    # Langues supportées
    "languages": ["fr", "en", "es", "de", "it"],
    "default_language": "fr",
}

# ==========================================================
# 📊 CONFIGURATION DES MÉTRIQUES (Évaluation)
# ==========================================================

METRICS_CONFIG = {
    # Activer le calcul de métriques
    "enable_metrics": False,  # Désactivé par défaut pour performance
    # Métriques pour résumés
    "summary_metrics": ["rouge", "bleu", "bertscore"],
    # Métriques pour quiz
    "quiz_metrics": ["diversity", "difficulty", "relevance"],
    # Seuils de qualité
    "quality_thresholds": {
        "min_summary_length": 20,  # mots
        "max_summary_length": 500,  # mots
        "min_question_length": 5,  # mots
        "min_options": 4,
    },
}

# ==========================================================
# 🚀 PRESETS (Configurations prédéfinies)
# ==========================================================

PRESETS = {
    "fast": {
        # Configuration optimisée pour vitesse
        "summarizer_model": "google/mt5-small",
        "quiz_model": "google/flan-t5-small",
        "num_beams": 2,
        "use_mixed_precision": True,
    },
    "balanced": {
        # Configuration équilibrée (par défaut)
        "summarizer_model": "facebook/mbart-large-50",
        "quiz_model": "google/flan-t5-base",
        "num_beams": 4,
        "use_mixed_precision": True,
    },
    "quality": {
        # Configuration optimisée pour qualité
        "summarizer_model": "facebook/mbart-large-50",
        "quiz_model": "google/flan-t5-large",
        "num_beams": 8,
        "use_mixed_precision": False,
    },
    "cpu_optimized": {
        # Configuration pour CPU (sans GPU)
        "summarizer_model": "google/mt5-base",
        "quiz_model": "google/flan-t5-base",
        "num_beams": 2,
        "use_mixed_precision": False,
        "num_threads": 8,
    },
}

# ==========================================================
# 🔒 CONFIGURATION DE SÉCURITÉ
# ==========================================================

SECURITY_CONFIG = {
    # Longueur maximale des entrées (protection contre DOS)
    "max_input_chars": 50000,  # ~10000 mots
    # Nombre maximum de questions par requête
    "max_questions_per_request": 50,
    # Rate limiting (si implémenté)
    "rate_limit_enabled": False,
    "requests_per_minute": 60,
}
