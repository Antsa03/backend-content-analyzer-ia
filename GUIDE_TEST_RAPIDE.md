# 🎯 Guide de Test Rapide

## ✅ Étape 1 : Vérifier les installations

```bash
# Vérifier que les packages sont installés
pip list | findstr "sentencepiece\|sacremoses\|protobuf"
```

Vous devriez voir :

```
protobuf          x.x.x
sacremoses        x.x.x
sentencepiece     x.x.x
```

## ✅ Étape 2 : Tester avec le script de test

```bash
python test_summary_quality.py
```

Ce script va :

1. 📊 Générer un résumé TF-IDF (classique)
2. 🧠 Générer un résumé avec BARThez (optimisé)
3. 🔍 Comparer les résultats

## ✅ Étape 3 : Tester avec l'API

### Démarrer le serveur

```bash
python main.py
```

### Tester un résumé neural (nouveau modèle)

```bash
# PowerShell
Invoke-RestMethod -Uri "http://localhost:8000/neural/summarize/text" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"text": "Votre texte ici...", "summary_length": "medium"}'
```

### Comparer TF-IDF vs Neural

```bash
# PowerShell
Invoke-RestMethod -Uri "http://localhost:8000/hybrid/summarize/text" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"text": "Votre texte ici...", "summary_length": "medium"}'
```

## 📊 Ce qui a changé

### Avant (mBART généraliste)

- ❌ Erreurs de tokenizer
- ❌ Résumés répétitifs
- ❌ Qualité moyenne en français

### Après (BARThez optimisé)

- ✅ Tokenizer stable
- ✅ Pas de répétitions
- ✅ Excellente qualité en français
- ✅ Résumés plus concis

## 🎛️ Paramètres Personnalisables

Dans `models/config.py`, vous pouvez ajuster :

```python
SUMMARIZER_CONFIG = {
    "model_name": "moussaKam/barthez-orangesum-abstract",  # Modèle FR optimisé
    "generation_params": {
        "medium": {
            "num_beams": 6,              # ⬆️ Augmenter pour plus de qualité
            "temperature": 0.9,          # ⬇️ Baisser pour plus de cohérence
            "repetition_penalty": 1.2,   # ⬆️ Augmenter contre répétitions
        }
    }
}
```

## 🔧 Résolution de Problèmes

### Le modèle ne se charge pas ?

```bash
# Vérifier les dépendances
pip install --upgrade transformers torch sentencepiece sacremoses protobuf
```

### Toujours des erreurs de tokenizer ?

Redémarrez l'application complètement :

1. Arrêter tous les processus Python
2. Relancer `python main.py`

### Qualité insuffisante ?

Essayez ces modèles alternatifs dans `neural_summarizer.py` :

```python
# Option 1 : T5 français (plus lourd mais très bon)
summarizer = model_manager.get_summarizer(
    model_name="plguillou/t5-base-fr-sum-cnndm"
)

# Option 2 : mT5 (plus léger, multilingue)
summarizer = model_manager.get_summarizer(
    model_name="google/mt5-base"
)
```

## 📈 Résultats Attendus

**Texte d'entrée** : ~200 mots sur l'IA

**Résumé Medium (40-150 mots)** :

- Temps : 2-5 secondes
- Qualité : Excellente en français
- Cohérence : Très bonne
- Répétitions : Aucune

## 💡 Prochaines Étapes

1. ✅ Tester avec `test_summary_quality.py`
2. ✅ Comparer avec vos propres textes
3. ✅ Ajuster les paramètres si besoin
4. ✅ Intégrer dans votre workflow

---

**Besoin d'aide ?** Consultez `AMELIORATIONS_RESUME.md` pour plus de détails !
