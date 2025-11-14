import PyPDF2, io
from .text_cleaner import clean_text_for_web

def extract_text_from_pdf(file_content: bytes) -> str:
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
    text = "\n".join([p.extract_text() or "" for p in pdf_reader.pages])
    return clean_text_for_web(text)
