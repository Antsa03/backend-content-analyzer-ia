from pydantic import BaseModel
from typing import List, Optional

class TextInput(BaseModel):
    text: str
    summary_length: Optional[str] = "medium"

class SummaryResponse(BaseModel):
    summary: str
    word_count: int
    original_length: int
