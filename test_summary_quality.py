"""
Script de test pour comparer la qualité des résumés
avant et après les améliorations.
"""

import sys
from utils.neural_summarizer import generate_summary_neural
from utils.summarizer import generate_summary

# Texte de test en français
TEST_TEXT = """
L'intelligence artificielle (IA) est un ensemble de théories et de techniques 
mises en œuvre en vue de réaliser des machines capables de simuler l'intelligence humaine. 
Elle repose sur des algorithmes sophistiqués et des réseaux de neurones artificiels qui 
permettent aux machines d'apprendre à partir de données. Les applications de l'IA sont 
nombreuses et variées : reconnaissance vocale, vision par ordinateur, traitement du 
langage naturel, voitures autonomes, diagnostic médical, et bien d'autres domaines. 

Les réseaux de neurones profonds, ou deep learning, constituent une branche majeure 
de l'IA moderne. Inspirés du fonctionnement du cerveau humain, ces réseaux sont composés 
de multiples couches de neurones artificiels qui traitent l'information de manière 
hiérarchique. Cette architecture permet aux modèles d'extraire automatiquement des 
caractéristiques complexes des données, sans nécessiter une programmation explicite 
de chaque règle.

L'apprentissage automatique, ou machine learning, est le processus par lequel 
les systèmes d'IA s'améliorent avec l'expérience. Au lieu d'être explicitement 
programmés pour chaque tâche, ces systèmes apprennent des patterns à partir de 
grandes quantités de données. Il existe plusieurs types d'apprentissage : supervisé, 
non supervisé, et par renforcement, chacun adapté à différents types de problèmes.

Les transformers représentent une architecture révolutionnaire en traitement du 
langage naturel. Introduits en 2017, ils utilisent un mécanisme d'attention qui 
permet au modèle de pondérer l'importance de différentes parties d'une séquence 
d'entrée. Cette innovation a conduit au développement de modèles de langage puissants 
comme GPT, BERT, et T5, capables de comprendre et de générer du texte avec une 
qualité remarquable.

Les défis éthiques et sociétaux liés à l'IA sont nombreux. Les questions de biais 
algorithmiques, de protection de la vie privée, d'impact sur l'emploi, et de contrôle 
des systèmes autonomes soulèvent des préoccupations importantes. Il est crucial de 
développer des cadres réglementaires et des principes éthiques pour guider le 
développement et le déploiement responsable de l'IA.
"""


def test_summaries():
    """Teste et compare les différentes méthodes de résumé."""
    print("=" * 80)
    print("🧪 TEST DE QUALITÉ DES RÉSUMÉS - SYSTÈME PAR POURCENTAGES")
    print("=" * 80)
    print()

    print("📝 TEXTE ORIGINAL")
    print("-" * 80)
    word_count = len(TEST_TEXT.split())
    print(f"Longueur : {word_count} mots")
    print(f"Extrait : {TEST_TEXT[:200]}...")
    print()

    print("📊 SYSTÈME DE RÉDUCTION PAR POURCENTAGES :")
    print("-" * 80)
    print("• SHORT (20%)  : conserve 20% du texte (réduction 80%)")
    print("• MEDIUM (40%) : conserve 40% du texte (réduction 60%)")
    print("• LONG (55%)   : conserve 55% du texte (réduction 45%)")
    print()

    # Calculs théoriques
    short_target = int(word_count * 0.20)
    medium_target = int(word_count * 0.40)
    long_target = int(word_count * 0.55)

    print(f"📐 LONGUEURS THÉORIQUES ATTENDUES (pour {word_count} mots) :")
    print(
        f"   • SHORT  : ~{short_target} mots (plage: {int(short_target*0.7)}-{int(short_target*1.3)} mots)"
    )
    print(
        f"   • MEDIUM : ~{medium_target} mots (plage: {int(medium_target*0.7)}-{int(medium_target*1.3)} mots)"
    )
    print(
        f"   • LONG   : ~{long_target} mots (plage: {int(long_target*0.7)}-{min(512, int(long_target*1.3))} mots)"
    )
    print()

    # Test 1 : Résumé TF-IDF (classique)
    print("=" * 80)
    print("📊 MÉTHODE 1 : TF-IDF (Extractif Classique)")
    print("=" * 80)
    try:
        tfidf_summary = generate_summary(TEST_TEXT, length="medium")
        tfidf_words = len(tfidf_summary.split())
        print(f"✅ Résumé généré ({tfidf_words} mots)")
        print(f"📄 Résultat :\n{tfidf_summary}")
    except Exception as e:
        print(f"❌ Erreur : {e}")
    print()

    # Test 2 : Résumé Neural SHORT
    print("=" * 80)
    print("🧠 MÉTHODE 2 : BARThez Neural - SHORT (20% du texte)")
    print("=" * 80)
    try:
        neural_short = generate_summary_neural(TEST_TEXT, length="short")
        short_words = len(neural_short.split())
        reduction_percent = (1 - short_words / word_count) * 100
        print(f"✅ Résumé généré ({short_words} mots)")
        print(f"📊 Réduction réelle : {reduction_percent:.1f}%")
        print(f"📄 Résultat :\n{neural_short}")
    except Exception as e:
        print(f"❌ Erreur : {e}")
    print()

    # Test 3 : Résumé Neural MEDIUM
    print("=" * 80)
    print("🧠 MÉTHODE 3 : BARThez Neural - MEDIUM (40% du texte) ⭐")
    print("=" * 80)
    try:
        neural_medium = generate_summary_neural(TEST_TEXT, length="medium")
        medium_words = len(neural_medium.split())
        reduction_percent = (1 - medium_words / word_count) * 100
        print(f"✅ Résumé généré ({medium_words} mots)")
        print(f"📊 Réduction réelle : {reduction_percent:.1f}%")
        print(f"📄 Résultat :\n{neural_medium}")
    except Exception as e:
        print(f"❌ Erreur : {e}")
    print()

    # Test 4 : Résumé Neural LONG
    print("=" * 80)
    print("🧠 MÉTHODE 4 : BARThez Neural - LONG (55% du texte)")
    print("=" * 80)
    try:
        neural_long = generate_summary_neural(TEST_TEXT, length="long")
        long_words = len(neural_long.split())
        reduction_percent = (1 - long_words / word_count) * 100
        print(f"✅ Résumé généré ({long_words} mots)")
        print(f"📊 Réduction réelle : {reduction_percent:.1f}%")
        print(f"📄 Résultat :\n{neural_long}")
    except Exception as e:
        print(f"❌ Erreur : {e}")
    print()

    # Analyse comparative
    print("=" * 80)
    print("📊 ANALYSE DU SYSTÈME PAR POURCENTAGES")
    print("=" * 80)
    print("✅ Avantages du système :")
    print("  • Adaptation automatique à la longueur du texte source")
    print("  • Ratios constants : short=20%, medium=40%, long=55%")
    print("  • Résumés proportionnels et cohérents")
    print("  • Flexibilité de ±30% pour s'adapter au contenu")
    print()
    print("📐 Formule appliquée :")
    print("  longueur_cible = longueur_source × pourcentage_conservation")
    print("  plage = [cible × 0.7, cible × 1.3]")
    print("  limites = [30 mots minimum, 512 mots maximum]")
    print()


if __name__ == "__main__":
    print()
    print("🚀 Lancement des tests de qualité des résumés...")
    print()
    test_summaries()
    print()
    print("=" * 80)
    print("✅ Tests terminés !")
    print("=" * 80)
    print()
    print("💡 Conseils :")
    print("   1. Comparez la fluidité des deux résumés")
    print("   2. Vérifiez qu'il n'y a pas de répétitions")
    print("   3. Testez avec vos propres textes")
    print("   4. Ajustez les paramètres dans models/config.py si nécessaire")
    print()
