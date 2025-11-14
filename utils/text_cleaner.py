import re, unicodedata

def clean_text_for_web(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = "".join(
        c for c in text if unicodedata.category(c)[0] != "C" or c in "\n\r\t"
    )
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\u00a0", " ")
    return text.strip()
