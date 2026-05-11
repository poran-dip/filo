from langchain_core.tools import tool
from server.drive import get_drive_service, search_files

_service = get_drive_service()

@tool
def drive_search(query: str) -> str:
    """
    Search Google Drive using a Drive API q-parameter string.
    Examples:
      name contains 'budget'
      mimeType = 'application/pdf'
      fullText contains 'quarterly report'
      modifiedTime > '2024-01-01T00:00:00' and name contains 'invoice'
    Combine clauses with 'and' / 'or'. Always return results as a list.
    """
    results = search_files(_service, query)
    if not results:
        return "No files found."
    return str(results)
