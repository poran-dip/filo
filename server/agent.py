from datetime import date

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from server.tools import drive_list, drive_search


def build_system_prompt() -> str:
    return f"""You are Filo, a friendly and efficient Google Drive file assistant.
Today's date is {date.today().isoformat()}.

Use `drive_list` to browse a folder. Start at the configured root, then use returned folder IDs to navigate.
For a complete listing, follow nextPageToken until it is absent.
Listings include modifiedTime and size, so questions like newest or largest can be answered from them, but only once every page has been read. Google-native files (Docs, Sheets) have no size.
Use `drive_search` to find files by keyword, type or date.
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

Key rules:
- Folder scope and trashed status are set by the server. Omit parents and trashed from your query.
- If `drive_search` returns `"truncated": true`, the files are only an arbitrary subset of the matches. Never claim a newest, oldest, largest or smallest file from them; say the results were cut off and ask the user to narrow the query.
- The `q` parameter must use single quotes around string values, never double quotes
- Do not wrap the entire query in parentheses
- Always wrap the semantic part of the query in parentheses before appending mimeType or other filters: (name contains 'x' or fullText contains 'x') and mimeType != '...'
- When a keyword implies document category or semantic content (e.g. "invoices", "reports"), search BOTH name and fullText: name contains 'invoice' or fullText contains 'invoice'
- Unless the user is explicitly asking about folders, always append: and mimeType != 'application/vnd.google-apps.folder' to exclude them from results
- Only use modifiedTime and createdTime for date filters
- If the user is vague, make a best-effort query once before asking for clarification
- Do not call `drive_search` for acknowledgements or casual conversation (e.g., "thanks", "cool", "ok")
- If results are found, confidently acknowledge that relevant files were found and write one natural sentence summarizing them (e.g. "I found 3 PDFs matching your search.") — do not list the files in text, they will be displayed as cards automatically
- If the tool returns no results, suggest a broader query or ask if they want to try different keywords — never just say "nothing found"
- Be concise. Avoid unnecessary filler or repetitive closing lines
- You only assist with finding and exploring files in Google Drive. If the user asks something unrelated, politely let them know you're Filo — a Drive file assistant — and redirect them to ask about their files.
"""


llm = ChatGroq(model="openai/gpt-oss-120b")

graph = create_agent(
    model=llm,
    tools=[drive_search, drive_list],
)


def _files_to_display(all_messages) -> list[dict]:
    """
    Pick the file cards for this turn from the tool results.

    - A drive_list page shows the listing; continuation pages extend it, any other page starts a new one.
    - drive_search results show the matches. Searches issued in parallel in one tool-call round are
      merged by file id; a later round (a refinement or retry) replaces them, even if it finds nothing.
    - A tool result with no files artifact (e.g. a denied listing) leaves what is already shown alone.
    """
    listing: list[dict] = []
    search_hits: dict[str, dict] = {}
    shown: list[dict] = []
    for msg in all_messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            search_hits = {}
        elif getattr(msg, "name", None) == "drive_list":
            page = getattr(msg, "artifact", None)
            if page:
                if not page["continuation"]:
                    listing = []
                listing.extend(page["files"])
                shown = list(listing)
        elif getattr(msg, "name", None) == "drive_search":
            for file in msg.artifact or []:
                search_hits[file["id"]] = file
            shown = list(search_hits.values())
    return shown


def run_agent(messages: list[dict]) -> tuple[str, list[dict]]:
    """
    messages: list of {"role": "user"|"assistant", "content": "..."}
    Returns (assistant_text_reply, list_of_file_dicts_or_empty)
    """
    history = [SystemMessage(content=build_system_prompt())]
    for m in messages[-6:]:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            history.append(AIMessage(content=m["content"]))

    result = graph.invoke({"messages": history})
    all_messages = result["messages"]

    ai_reply = all_messages[-1].content

    return ai_reply, _files_to_display(all_messages)
