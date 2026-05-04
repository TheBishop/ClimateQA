from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.api.chain import ask
import time

app = FastAPI(
    title="ClimateQA API",
    description="Query IPCC AR6 climate reports using natural language.",
    version="1.0.0",
)

class QuestionRequest(BaseModel):
    question: str

class SourceItem(BaseModel):
    file: str
    page: int | str
    snippet: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]
    latency_ms: float

@app.get("/")
def root():
    return {"message": "ClimateQA API is running"}

@app.get("/health")
def health():
    return {"status": "ok", "model": "llama-3.3-70b-versatile"}

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    start = time.time()
    result = ask(request.question)
    latency_ms = round((time.time() - start) * 1000, 2)
    return AnswerResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        latency_ms=latency_ms,
    )