"""
Module de détection et suppression des en-têtes de documents
Identifie et retire automatiquement les en-têtes répétitifs ou non pertinents
"""

import re
from typing import List, Tuple


def detect_and_remove_header(text: str) -> str:
    """
    Détecte et supprime intelligemment l'en-tête d'un document.

    Stratégies de détection :
    1. Lignes courtes répétées en haut du document
    2. Numéros de page, dates, informations métadonnées
    3. Titres de document génériques
    4. Sections avant le début du contenu principal

    Args:
        text: Le texte complet du document

    Returns:
        Le texte sans l'en-tête détecté
    """
    if not text or len(text.strip()) < 100:
        return text

    lines = text.split("\n")
    if len(lines) < 5:
        return text

    # Détection de l'en-tête
    header_end_index = _find_header_end(lines)

    # Si un en-tête est détecté, le retirer
    if header_end_index > 0:
        cleaned_lines = lines[header_end_index:]
        return "\n".join(cleaned_lines).strip()

    return text


def _find_header_end(lines: List[str]) -> int:
    """
    Trouve l'index de fin de l'en-tête dans une liste de lignes.

    Returns:
        L'index de la première ligne de contenu principal (0 si pas d'en-tête)
    """
    header_indicators = 0
    potential_end = 0

    # Analyser les 20 premières lignes (ou moins si le document est court)
    max_lines_to_check = min(20, len(lines))

    for i in range(max_lines_to_check):
        line = lines[i].strip()

        # Ignorer les lignes vides
        if not line:
            continue

        # Critères d'identification d'en-tête
        is_header_line = False

        # 1. Lignes très courtes (< 5 mots) en haut du document
        word_count = len(line.split())
        if i < 5 and word_count < 5 and word_count > 0:
            is_header_line = True

        # 2. Numéros de page (Page 1, p.1, etc.)
        if re.search(r"\b(page|p\.?)\s*\d+\b", line, re.IGNORECASE):
            is_header_line = True

        # 3. Dates en début de document
        if i < 8 and re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", line):
            is_header_line = True

        # 4. Informations de copyright, confidentialité
        if re.search(
            r"\b(copyright|©|confidentiel|confidential|propriété)\b",
            line,
            re.IGNORECASE,
        ):
            is_header_line = True

        # 5. En-têtes de document génériques
        generic_headers = [
            r"^(document|rapport|mémoire|thèse|article)\b",
            r"^(titre|title|sujet|subject)\s*:",
            r"^(auteur|author|par|by)\s*:",
            r"^(date|version)\s*:",
        ]
        for pattern in generic_headers:
            if re.search(pattern, line, re.IGNORECASE):
                is_header_line = True
                break

        # 6. Lignes en majuscules (souvent des titres d'en-tête)
        if i < 5 and line.isupper() and 3 <= word_count <= 10:
            is_header_line = True

        # 7. URLs, emails, numéros de téléphone
        if re.search(r"(https?://|www\.|@[\w\.-]+\.[\w]+|\+?\d{10,})", line):
            is_header_line = True

        # Comptage des indicateurs d'en-tête
        if is_header_line:
            header_indicators += 1
            potential_end = i + 1
        else:
            # Si on trouve une ligne substantielle (>= 10 mots), c'est probablement le début du contenu
            if word_count >= 10:
                # Vérifier si on a accumulé assez d'indicateurs d'en-tête
                if header_indicators >= 2:
                    return potential_end
                # Sinon, considérer que le contenu commence ici
                if i > 0:
                    return potential_end if potential_end > 0 else 0
                return 0

    # Si on a trouvé des indicateurs d'en-tête, retourner la position
    if header_indicators >= 2:
        return potential_end

    return 0


def remove_page_numbers(text: str) -> str:
    """
    Supprime les numéros de page du texte.

    Args:
        text: Le texte contenant potentiellement des numéros de page

    Returns:
        Le texte sans les numéros de page
    """
    # Motifs de numéros de page courants
    patterns = [
        r"\n\s*Page\s+\d+\s*\n",
        r"\n\s*p\.\s*\d+\s*\n",
        r"\n\s*-\s*\d+\s*-\s*\n",
        r"\n\s*\[\d+\]\s*\n",
        r"\n\s*\d+\s*/\s*\d+\s*\n",  # 1/10, 2/10, etc.
    ]

    cleaned_text = text
    for pattern in patterns:
        cleaned_text = re.sub(pattern, "\n", cleaned_text, flags=re.IGNORECASE)

    return cleaned_text


def remove_headers_and_footers(text: str) -> str:
    """
    Supprime les en-têtes et pieds de page répétitifs d'un document.
    Utile pour les PDFs extraits qui conservent ces éléments.

    Args:
        text: Le texte du document

    Returns:
        Le texte nettoyé
    """
    lines = text.split("\n")

    if len(lines) < 10:
        return text

    # Détecter les lignes répétitives (en-têtes/pieds de page)
    line_frequency = {}
    for line in lines:
        cleaned = line.strip()
        if cleaned and len(cleaned) < 100:  # Les en-têtes sont généralement courts
            line_frequency[cleaned] = line_frequency.get(cleaned, 0) + 1

    # Identifier les lignes qui se répètent plus de 2 fois (probablement en-têtes/pieds de page)
    repetitive_lines = {
        line
        for line, count in line_frequency.items()
        if count >= 3 and len(line.split()) <= 15
    }

    # Filtrer ces lignes
    filtered_lines = [
        line
        for line in lines
        if line.strip() not in repetitive_lines or len(line.strip()) == 0
    ]

    return "\n".join(filtered_lines)


def smart_clean_document(text: str) -> str:
    """
    Nettoyage intelligent complet d'un document :
    - Suppression de l'en-tête
    - Suppression des numéros de page
    - Suppression des en-têtes/pieds de page répétitifs

    Args:
        text: Le texte brut du document

    Returns:
        Le texte nettoyé et prêt pour l'analyse
    """
    if not text or len(text.strip()) < 50:
        return text

    # Étape 1 : Supprimer l'en-tête principal
    text = detect_and_remove_header(text)

    # Étape 2 : Supprimer les numéros de page
    text = remove_page_numbers(text)

    # Étape 3 : Supprimer les en-têtes/pieds de page répétitifs
    text = remove_headers_and_footers(text)

    # Étape 4 : Nettoyer les espaces multiples
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)  # Max 2 sauts de ligne consécutifs
    text = re.sub(r" +", " ", text)  # Espaces multiples → un seul espace

    return text.strip()
