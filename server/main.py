from fastapi import FastAPI
from pydantic import BaseModel

from server.drive import get_drive_service, search_files
from server.groq import ask_groq

app = FastAPI()

service = get_drive_service()


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/chat")
async def chat(body: ChatRequest):
    user_msg = body.message
    files = search_files(service, f"name contains '{user_msg}'")
    # res = ask_groq(body.message)
    return {
        "response": f"Found {len(files)} files",
    }
