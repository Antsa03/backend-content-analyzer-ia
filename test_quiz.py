"""
Script de test pour la génération de quiz
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

from utils.neural_quiz_generator import generate_quiz_neural

# Texte de test en français
test_text = """
L'intelligence artificielle (IA) est un ensemble de théories et de techniques visant à réaliser 
des machines capables de simuler l'intelligence humaine. Elle englobe plusieurs domaines comme 
l'apprentissage automatique (machine learning), le traitement du langage naturel et la vision par ordinateur.

Le machine learning permet aux ordinateurs d'apprendre à partir de données sans être explicitement programmés. 
Les réseaux de neurones artificiels s'inspirent du fonctionnement du cerveau humain avec des couches de neurones 
interconnectés. L'apprentissage profond (deep learning) utilise des réseaux de neurones avec plusieurs couches cachées.

Python est le langage de programmation le plus populaire pour l'IA grâce à ses bibliothèques comme TensorFlow, 
PyTorch et scikit-learn. Ces frameworks facilitent le développement de modèles d'apprentissage automatique.

Les applications de l'IA sont nombreuses: reconnaissance vocale, traduction automatique, voitures autonomes, 
diagnostic médical, recommandations personnalisées. L'IA transforme profondément de nombreux secteurs d'activité.
"""

print("🧪 Test de génération de quiz en français...")
print("=" * 80)

try:
    questions = generate_quiz_neural(test_text, num_questions=5)

    print(f"\n✅ {len(questions)} questions générées:")
    print("=" * 80)

    for i, q in enumerate(questions, 1):
        print(f"\n❓ Question {i}: {q['question']}")
        print(f"   Options:")
        for j, opt in enumerate(q["options"]):
            marker = "✓" if j == q["correct_answer"] else " "
            print(f"      [{marker}] {j+1}. {opt}")
        print(f"   💡 Explication: {q['explanation'][:100]}...")

    if not questions:
        print("\n❌ AUCUNE QUESTION GÉNÉRÉE!")
        print("Vérifiez les logs ci-dessus pour voir ce qui bloque.")

except Exception as e:
    print(f"\n❌ ERREUR: {e}")
    import traceback

    traceback.print_exc()
