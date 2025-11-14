from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models.schemas import TextInput, SummaryResponse
from utils.pdf_utils import extract_text_from_pdf
from utils.summarizer import generate_summary
from utils.quiz_generator import generate_quiz
from utils.text_cleaner import clean_text_for_web

# Import des nouveaux modules Deep Learning
from utils.neural_summarizer import generate_summary_neural, generate_summary_hybrid
from utils.neural_quiz_generator import generate_quiz_neural, generate_quiz_hybrid

import uvicorn
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Content Analyzer with Deep Learning",
    version="3.0",
    description="API utilisant des réseaux de neurones Transformer pour résumer et générer des quiz",
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "API d'analyse de contenu avec Deep Learning (v3.0)",
        "description": "Utilise des réseaux de neurones Transformer (mBART, T5, FLAN-T5)",
        "endpoints": {
            "Classique (TF-IDF)": {
                "/summarize/text": "POST - Résumer du texte (TF-IDF)",
                "/summarize/pdf": "POST - Résumer un PDF (TF-IDF)",
                "/generate-quiz/text": "POST - Générer un quiz depuis du texte (extractif)",
                "/generate-quiz/pdf": "POST - Générer un quiz depuis un PDF (extractif)",
            },
            "Deep Learning (Réseaux de neurones)": {
                "/neural/summarize/text": "POST - Résumer avec réseau de neurones",
                "/neural/summarize/pdf": "POST - Résumer PDF avec réseau de neurones",
                "/neural/generate-quiz/text": "POST - Quiz avec réseau de neurones",
                "/neural/generate-quiz/pdf": "POST - Quiz PDF avec réseau de neurones",
            },
            "Hybride (Comparaison)": {
                "/hybrid/summarize/text": "POST - Compare TF-IDF vs Deep Learning",
                "/hybrid/generate-quiz/text": "POST - Compare génération classique vs neuronale",
            },
        },
        "technologies": {
            "framework": "FastAPI",
            "ml_models": ["mBART (Facebook)", "T5 (Google)", "FLAN-T5"],
            "architecture": "Transformer (Encoder-Decoder)",
            "deep_learning": "PyTorch + Hugging Face Transformers",
        },
    }


@app.post("/summarize/text", response_model=SummaryResponse)
async def summarize_text(input_data: TextInput):
    text = clean_text_for_web(input_data.text)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide.")

    if len(text.split()) < 10:
        raise HTTPException(status_code=400, detail="Le texte est trop court.")

    summary = generate_summary(text, input_data.summary_length)
    return {
        "summary": summary,
        "word_count": len(summary.split()),
        "original_length": len(text.split()),
    }


@app.post("/summarize/pdf", response_model=SummaryResponse)
async def summarize_pdf(file: UploadFile = File(...), length: str = "medium"):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Le fichier doit être un PDF.")
    content = await file.read()

    text = extract_text_from_pdf(content)
    summary = generate_summary(text, length)

    return {
        "summary": summary,
        "word_count": len(summary.split()),
        "original_length": len(text.split()),
    }


@app.post("/generate-quiz/text")
async def create_quiz_from_text(input_data: TextInput):
    text = clean_text_for_web(input_data.text)
    if len(text.split()) < 100:
        raise HTTPException(
            status_code=400,
            detail="Le texte doit contenir au moins 100 mots pour générer un quiz.",
        )

    questions = generate_quiz(text, num_questions=20)
    return {"questions": questions, "topic": "Contenu fourni"}


@app.post("/generate-quiz/pdf")
async def create_quiz_from_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Le fichier doit être un PDF.")
    content = await file.read()

    text = extract_text_from_pdf(content)
    questions = generate_quiz(text, num_questions=20)
    topic = clean_text_for_web(file.filename.rsplit(".", 1)[0])

    return {"questions": questions, "topic": topic}


# ==========================================================
# 🧠 ENDPOINTS DEEP LEARNING (Réseaux de neurones)
# ==========================================================


@app.post("/neural/summarize/text", response_model=SummaryResponse)
async def neural_summarize_text(input_data: TextInput):
    """
    Résume du texte en utilisant un réseau de neurones Transformer (mBART).
    Architecture: Encoder-Decoder avec attention multi-têtes.
    """
    text = clean_text_for_web(input_data.text)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide.")

    if len(text.split()) < 10:
        raise HTTPException(status_code=400, detail="Le texte est trop court.")

    logger.info("🧠 Génération de résumé avec réseau de neurones...")
    summary = generate_summary_neural(text, input_data.summary_length)

    return {
        "summary": summary,
        "word_count": len(summary.split()),
        "original_length": len(text.split()),
    }


@app.post("/neural/summarize/pdf", response_model=SummaryResponse)
async def neural_summarize_pdf(file: UploadFile = File(...), length: str = "medium"):
    """
    Résume un PDF en utilisant un réseau de neurones Transformer.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Le fichier doit être un PDF.")
    content = await file.read()

    text = extract_text_from_pdf(content)

    logger.info("🧠 Génération de résumé PDF avec réseau de neurones...")
    summary = generate_summary_neural(text, length)

    return {
        "summary": summary,
        "word_count": len(summary.split()),
        "original_length": len(text.split()),
    }


@app.post("/neural/generate-quiz/text")
async def neural_create_quiz_from_text(input_data: TextInput):
    """
    Génère un quiz en utilisant un réseau de neurones T5/FLAN-T5.
    Architecture: Transformer avec génération de questions intelligentes.
    """
    text = clean_text_for_web(input_data.text)
    if len(text.split()) < 100:
        raise HTTPException(
            status_code=400,
            detail="Le texte doit contenir au moins 100 mots pour générer un quiz.",
        )

    logger.info("🧠 Génération de quiz avec réseau de neurones...")
    questions = generate_quiz_neural(text, num_questions=20)

    return {
        "questions": questions,
        "topic": "Contenu fourni",
        "generation_method": "neural_network",
    }


@app.post("/neural/generate-quiz/pdf")
async def neural_create_quiz_from_pdf(file: UploadFile = File(...)):
    """
    Génère un quiz depuis un PDF en utilisant un réseau de neurones.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Le fichier doit être un PDF.")
    content = await file.read()

    text = extract_text_from_pdf(content)

    logger.info("🧠 Génération de quiz PDF avec réseau de neurones...")
    questions = generate_quiz_neural(text, num_questions=20)
    topic = clean_text_for_web(file.filename.rsplit(".", 1)[0])

    return {
        "questions": questions,
        "topic": topic,
        "generation_method": "neural_network",
    }


# ==========================================================
# 🔀 ENDPOINTS HYBRIDES (Comparaison TF-IDF vs Deep Learning)
# ==========================================================


@app.post("/hybrid/summarize/text")
async def hybrid_summarize_text(input_data: TextInput):
    """
    Compare les résumés générés par TF-IDF et par réseau de neurones.
    Permet de voir la différence entre approche extractive et abstractive.
    """
    text = clean_text_for_web(input_data.text)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Le texte ne peut pas être vide.")

    logger.info("🔀 Génération de résumé hybride (TF-IDF + Neural)...")
    result = generate_summary_hybrid(text, input_data.summary_length)

    return {
        "neural_summary": result["neural_summary"],
        "tfidf_summary": result["tfidf_summary"],
        "neural_word_count": len(result["neural_summary"].split()),
        "tfidf_word_count": len(result["tfidf_summary"].split()),
        "original_length": len(text.split()),
        "comparison": "Le résumé neural est abstractif (reformulation), le TF-IDF est extractif (sélection de phrases)",
    }


@app.post("/hybrid/generate-quiz/text")
async def hybrid_create_quiz_from_text(input_data: TextInput):
    """
    Compare les quiz générés par l'approche classique et par réseau de neurones.
    """
    text = clean_text_for_web(input_data.text)
    if len(text.split()) < 100:
        raise HTTPException(
            status_code=400,
            detail="Le texte doit contenir au moins 100 mots pour générer un quiz.",
        )

    logger.info("🔀 Génération de quiz hybride (Classique + Neural)...")
    result = generate_quiz_hybrid(text, num_questions=20)

    return {
        "neural_quiz": result["neural_quiz"],
        "classic_quiz": result["classic_quiz"],
        "neural_count": len(result["neural_quiz"]),
        "classic_count": len(result["classic_quiz"]),
        "comparison": "Le quiz neural génère des questions plus naturelles, le classique est basé sur l'extraction",
    }


# ==========================================================
# 🚀 Fonction main : permet de lancer directement le serveur
# ==========================================================
def main():
    """
    Point d'entrée principal pour exécuter l’API sans passer par uvicorn en ligne de commande.
    Exemple : python main.py
    """
    print("🚀 Démarrage de l’API AI Content Analyzer...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


# ==========================================================
# ✅ Point d'exécution
# ==========================================================
if __name__ == "__main__":
    main()
