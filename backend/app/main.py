from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vector_search import search_similar_questions, generate_answer, extract_category, construct_messages
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
    messages = construct_messages(query.question, similar_QAs)
    results = generate_answer(messages)
    category = extract_category(similar_QAs)

    return {
        "question": query.question,
        "answer": results,
        "category": category,
    }


@app.post("/stream")
def stream_response(req: Query):
    def token_stream():
        similar_QAs = search_similar_questions(req.question)
        messages = construct_messages(req.question, similar_QAs)
        category = extract_category(similar_QAs)

        yield f"[[CATEGORY:{category}]]\n"

        for chunk in openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True,
        ):
            if "choices" in chunk:
                yield chunk["choices"][0]["delta"].get("content", "")

    return StreamingResponse(token_stream(), media_type="text/plain")