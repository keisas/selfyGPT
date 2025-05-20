from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vector_search import search_similar_questions, generate_answer, extract_category
import openai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # セキュリティ目的で限定も可能
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    similar_QAs = search_similar_questions(query.question)
    results = generate_answer(query.question, similar_QAs)
    category = extract_category(similar_QAs)

    return {
        "question": query.question,
        "answer": results,
        "category": category,
    }