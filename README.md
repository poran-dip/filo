# Filo AI

Filo AI is a conversational Drive assistant that finds your files through natural chat. Search by name, type, content, or date — Filo translates your intent into precise Drive API queries and stays focused on your files.

## Stack

- **Backend** — Python, FastAPI, LangGraph, LangChain, Groq (Llama 3.3 70B)
- **Frontend** — Streamlit
- **Integration** — Google Drive API v3 via Service Account

## Features

- Natural language search across a designated Google Drive folder (and all subfolders)
- Searches by file name, MIME type, full-text content, and date filters
- Multi-turn conversation — follow-up and refine searches naturally across the session
- File results rendered as cards with direct Drive links
- Powered by LangGraph tool-calling — the LLM constructs valid Drive `q` parameter strings
- Stays focused — politely redirects off-topic questions back to file search
- Smart no-results handling — suggests broader queries instead of dead ends

## Project Structure

```
filo/
├── server/
│   ├── main.py       # FastAPI app
│   ├── agent.py      # LangGraph agent + system prompt
│   ├── tools.py      # drive_search LangChain tool
│   └── drive.py      # Google Drive API client
├── client/
│   └── app.py        # Streamlit chat interface
├── render.yaml       # Render deployment config
├── requirements.txt
└── .env.example
```

## Local Setup

### Prerequisites

- Python 3.12+
- A Google Cloud Service Account with Drive API enabled
- The service account shared on your target Drive folder

### Steps

1. Clone the repo and create a virtual environment:

```bash
git clone https://github.com/your-username/drive-chat-agent
cd drive-chat-agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

3. Run the backend:

```bash
uvicorn server.main:app --reload
```

4. In a separate terminal, run the frontend:

```bash
streamlit run client/app.py
```

## Environment Variables

| Variable                      | Description                                                                 |
| ----------------------------- | --------------------------------------------------------------------------- |
| `GROQ_API_KEY`                | Your Groq API key                                                           |
| `DRIVE_FOLDER_ID`             | Root Google Drive folder ID to search within                                |
| `GOOGLE_SERVICE_ACCOUNT_JSON` | Full contents of the service account JSON file (single line)                |
| BACKEND_URL                   | URL of the deployed backend (e.g. https://drive-agent-backend.onrender.com) |

See `.env.example` for reference.

## Deployment

- **Backend** — Render (configured via `render.yaml`). Set the three env vars in the Render dashboard.
- **Frontend** — Streamlit Community Cloud. Connect the GitHub repo, set main file to `client/app.py`, and add `BACKEND_URL` pointing to your Render service URL in the app secrets.
