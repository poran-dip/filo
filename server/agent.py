from datetime import date
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.agents import create_agent

from server.tools import drive_search

SYSTEM_PROMPT = f"""You are a helpful Google Drive file assistant.
Today's date is {date.today().isoformat()}.

When the user asks to find, search, or filter files, you MUST call the `drive_search` tool.
Translate their natural language request into a valid Google Drive API `q` parameter string.

Drive q-string syntax rules:
- Name search:        name contains 'keyword'
- Exact name:         name = 'exact filename'
- File type:          mimeType = 'application/pdf'
                      mimeType = 'application/vnd.google-apps.spreadsheet'
                      mimeType = 'application/vnd.google-apps.document'
                      mimeType = 'image/jpeg' or mimeType contains 'image/'
- Full text search:   fullText contains 'keyword'
- Date filter:        modifiedTime > '2025-05-08T00:00:00'
- Combine:            name contains 'report' and modifiedTime > '2026-02-01T00:00:00'

After getting results, present them clearly. If no files are found, say so and suggest refining the search.
Never fabricate file names. Only report what the tool returns.
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
            import json
            try:
                import ast
                files = ast.literal_eval(msg.content)
            except Exception:
                files = []

    return ai_reply, files
