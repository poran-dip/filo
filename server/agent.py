import ast
from datetime import date
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from server.config import DRIVE_FOLDER_ID
from server.drive import get_drive_service, get_all_folder_ids
from server.tools import drive_search

_service = get_drive_service()
_folder_map = get_all_folder_ids(_service, DRIVE_FOLDER_ID)
_folder_list = "\n".join(f"  - {name}" for name in _folder_map if name != "root")

SYSTEM_PROMPT = f"""You are Filo, a friendly and efficient Google Drive file assistant.
Today's date is {date.today().isoformat()}.

When the user asks to find, search, or filter files, you MUST call the `drive_search` tool.
Translate their natural language request into a valid Google Drive API `q` parameter string.

Drive q-string syntax rules:
- Name search:        name contains 'keyword'
- Exact name:         name = 'exact filename'
- File type:          mimeType = 'application/pdf'
                      mimeType = 'application/vnd.google-apps.spreadsheet'
                      mimeType = 'application/vnd.google-apps.document'
                      mimeType contains 'image/'    (catches jpg, png, jpeg, etc.)
- Full text search:   fullText contains 'keyword'
- Date filters:       modifiedTime > '2024-01-01T00:00:00'
                      createdTime < '2024-06-01T00:00:00'
                      Operators: > <
- Combinators:        and, or, not, and parentheses for grouping
                      e.g. (name contains 'invoice' or fullText contains 'invoice') and mimeType = 'application/pdf'
                      e.g. not mimeType = 'application/pdf'

Available folders in this Drive:
{_folder_list}

Key rules:
- When a keyword implies document category or semantic content (e.g. "invoices", "reports"), search BOTH name and fullText: name contains 'invoice' or fullText contains 'invoice'
- Unless the user is explicitly asking about folders, always append: and mimeType != 'application/vnd.google-apps.folder' to exclude them from results
- Only use modifiedTime and createdTime for date filters
- If the user is vague, make a best-effort query once before asking for clarification
- If the tool returns no results, suggest a broader query or ask if they want to try different keywords — never just say "nothing found"
- When results are found, write one natural sentence summarizing them (e.g. "I found 3 PDFs matching your search.") — do not list the files in text, they will be displayed as cards automatically
- Be concise. Avoid unnecessary filler or repetitive closing lines
- You only assist with finding and exploring files in Google Drive. If the user asks something unrelated, politely let them know you're Filo — a Drive file assistant — and redirect them to ask about their files.
"""

llm = ChatGroq(model="llama-3.3-70b-versatile")

graph = create_agent(
    model=llm,
    tools=[drive_search],
    system_prompt=SystemMessage(content=SYSTEM_PROMPT),
)


def run_agent(messages: list[dict]) -> tuple[str, list[dict]]:
    """
    messages: list of {"role": "user"|"assistant", "content": "..."}
    Returns (assistant_text_reply, list_of_file_dicts_or_empty)
    """
    history = []
    for m in messages[-6:]:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            history.append(AIMessage(content=m["content"]))

    result = graph.invoke({"messages": history})
    all_messages = result["messages"]

    ai_reply = all_messages[-1].content

    files = []
    for msg in all_messages:
        if hasattr(msg, "name") and msg.name == "drive_search":
            if msg.content != "No files found.":
                try:
                    files = ast.literal_eval(msg.content)
                except Exception:
                    files = []

    return ai_reply, files
