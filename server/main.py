from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/chat")
async def chat(body: ChatRequest):
    return {
        "response": f"You said: {body.message}"
    }
