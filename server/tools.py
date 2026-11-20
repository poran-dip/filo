from googleapiclient.errors import HttpError
from langchain_core.tools import tool
from server.drive import get_drive_service, search_files
from server.query import InvalidDriveQuery

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
    try:
        results = search_files(_service, query)
    except InvalidDriveQuery as exc:
        return f"Invalid search filter: {exc} Revise the query and try again."
    except HttpError as exc:
        if exc.resp.status != 400:
            raise
        return f"Drive rejected the filter: {exc.reason}. Check field values and try again."
    if not results:
        return "No files found."
    return str(results)
