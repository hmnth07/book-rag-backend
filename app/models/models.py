# app/models.py
from pydantic import BaseModel
from typing import List

class QuestionRequest(BaseModel):
    question: str

class Chunk(BaseModel):
    id: int
    text: str

class AnswerResponse(BaseModel):
    answer: str
    chunks: List[Chunk]
