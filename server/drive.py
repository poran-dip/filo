import time
from google.oauth2 import service_account
from googleapiclient.discovery import build
from server.config import DRIVE_FOLDER_ID, SERVICE_ACCOUNT_INFO

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

_folder_cache: dict[str, str] | None = None
_cache_time: float = 0
CACHE_TTL = 3600    # 1 hour


def get_drive_service():
    creds = service_account.Credentials.from_service_account_info(
        SERVICE_ACCOUNT_INFO,
        scopes=SCOPES
    )

    return build("drive", "v3", credentials=creds)


def get_all_folder_ids(service, root_folder_id: str) -> dict[str, str]:
    global _folder_cache, _cache_time

    if _folder_cache and (time.time() - _cache_time) < CACHE_TTL:
        return _folder_cache

    folder_map = {"root": root_folder_id}
    queue = [root_folder_id]

    while queue:
        current = queue.pop()
        results = service.files().list(
            q=f"'{current}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false",
            fields="files(id, name)"
        ).execute()
        for folder in results.get("files", []):
            folder_map[folder["name"]] = folder["id"]
            queue.append(folder["id"])

    _folder_cache = folder_map
    _cache_time = time.time()
    return _folder_cache


def search_files(service, query: str) -> list[dict]:
    folder_map = get_all_folder_ids(service, DRIVE_FOLDER_ID)

    parent_clause = " or ".join(f"'{fid}' in parents" for fid in folder_map.values())
    q = f"({parent_clause}) and ({query}) and trashed = false"

    results = service.files().list(
        q=q,
        fields="files(id, name, mimeType, size, modifiedTime, webViewLink)"
    ).execute()

    return results.get("files", [])
