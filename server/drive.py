import time
from itertools import batched

from google.oauth2 import service_account
from googleapiclient.discovery import build

from server.config import DRIVE_FOLDER_ID, SERVICE_ACCOUNT_INFO
from server.query import validate_query

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

MAX_SEARCH_RESULTS = 100
PARENTS_PER_QUERY = 20
_folder_cache: tuple[str, float, tuple[str, ...]] | None = None
# A cached tree can miss new folders or retain moved folders until refresh.
CACHE_TTL = 60  # seconds


def get_drive_service():
    creds = service_account.Credentials.from_service_account_info(
        SERVICE_ACCOUNT_INFO, scopes=SCOPES
    )

    return build("drive", "v3", credentials=creds)


def _iter_files(service, query: str, fields: str):
    token = None
    while True:
        page = (
            service.files()
            .list(
                q=query,
                fields=f"nextPageToken,files({fields})",
                pageToken=token,
            )
            .execute()
        )
        yield from page.get("files", [])
        token = page.get("nextPageToken")
        if not token:
            return


def get_all_folder_ids(service, root_folder_id: str) -> tuple[str, ...]:
    global _folder_cache
    cached = _folder_cache
    if cached is not None:
        root, expires_at, ids = cached
        if root == root_folder_id and time.monotonic() < expires_at:
            return ids

    seen = {root_folder_id}
    pending = [root_folder_id]
    folder_ids = []
    while pending:
        current = pending.pop()
        folder_ids.append(current)
        query = (
            f"'{current}' in parents and mimeType = 'application/vnd.google-apps.folder'"
            " and trashed = false"
        )
        for folder in _iter_files(service, query, "id"):
            child = folder["id"]
            if child not in seen:
                seen.add(child)
                pending.append(child)

    ids = tuple(folder_ids)
    _folder_cache = (root_folder_id, time.monotonic() + CACHE_TTL, ids)
    return ids


def search_files(service, query: str) -> tuple[list[dict], bool]:
    """Return unique matches and whether an additional match was found."""
    validate_query(query)
    folder_ids = get_all_folder_ids(service, DRIVE_FOLDER_ID)
    files_by_id = {}
    for batch in batched(folder_ids, PARENTS_PER_QUERY):
        parents = " or ".join(f"'{fid}' in parents" for fid in batch)
        scoped_query = f"({parents}) and ({query}) and trashed = false"
        for file in _iter_files(
            service, scoped_query, "id,name,mimeType,size,modifiedTime,webViewLink"
        ):
            if len(files_by_id) == MAX_SEARCH_RESULTS and file["id"] not in files_by_id:
                return list(files_by_id.values()), True
            files_by_id[file["id"]] = file
    return list(files_by_id.values()), False
