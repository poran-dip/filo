import json
import re
from typing import Literal

from googleapiclient.errors import HttpError
from langchain_core.tools import tool
from server.config import DRIVE_FOLDER_ID
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


@tool(response_format="content_and_artifact")
def drive_list(
    folder_id: str | None = None,
    kind: Literal["all", "files", "folders"] = "all",
    page_token: str | None = None,
) -> tuple[str, dict]:
    """List one page of direct children. Omit folder_id for the configured root.
    For another page, reuse folder_id and kind with the returned nextPageToken.
    """
    folder = DRIVE_FOLDER_ID if folder_id is None else folder_id
    folder_type = "application/vnd.google-apps.folder"
    denied = "Choose a folder inside the configured root."
    if not re.fullmatch(r"[A-Za-z0-9_-]+", folder):
        return denied, {}
    try:
        # Check current ancestry on each call, including continuation pages.
        current = folder
        visited = set()
        while current != DRIVE_FOLDER_ID:
            if current in visited:
                return denied, {}
            visited.add(current)
            item = _service.files().get(
                fileId=current, fields="id,mimeType,parents,trashed"
            ).execute()
            if (item["id"] != current or item.get("trashed")
                    or item["mimeType"] != folder_type or not item.get("parents")):
                return denied, {}
            current = item["parents"][0]

        query = f"'{folder}' in parents and trashed = false"
        if kind != "all":
            operator = "=" if kind == "folders" else "!="
            query += f" and mimeType {operator} '{folder_type}'"
        page = _service.files().list(
            q=query, pageSize=50, pageToken=page_token,
            fields="nextPageToken,files(id,name,mimeType,size,modifiedTime,webViewLink)",
        ).execute()
    except HttpError as exc:
        if exc.resp.status in (403, 404):
            return denied, {}
        if exc.resp.status == 400:
            return "Invalid listing request. Retry from the first page.", {}
        raise
    return json.dumps(page), {"files": page.get("files", []), "continuation": bool(page_token)}
