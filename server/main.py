from fastapi import FastAPI
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
    reply, files = run_agent(body.messages)
    return {
        "response": reply,
        "files": files,
    }
