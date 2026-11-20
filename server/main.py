from fastapi import FastAPI
from groq import RateLimitError
from pydantic import BaseModel

from server.agent import run_agent

app = FastAPI()


class ChatRequest(BaseModel):
    messages: list[dict]


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/chat")
async def chat(body: ChatRequest):
    try:
        reply, files = run_agent(body.messages)
        return {
            "response": reply,
            "files": files,
        }
    except RateLimitError:
        return {
            "response": "I'm currently rate limited — please try again in a few minutes.",
            "files": [],
        }
