from fastapi import FastAPI
from pydantic import BaseModel

from server.groq import ask_groq

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/chat")
async def chat(body: ChatRequest):
    res = ask_groq(body.message)
    return {
        "response": res
    }
