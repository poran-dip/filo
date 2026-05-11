import os
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build

load_dotenv()
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


import os
import time

_folder_cache: list[str] | None = None
_cache_time: float = 0
CACHE_TTL = 3600    # 1 hour


def get_drive_service():
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not set")
    creds = service_account.Credentials.from_service_account_file(
        creds_path,
        scopes=SCOPES
    )

    return build("drive", "v3", credentials=creds)


def get_all_folder_ids(service, root_folder_id: str) -> list[str]:
    global _folder_cache, _cache_time

    if _folder_cache and (time.time() - _cache_time) < CACHE_TTL:
        return _folder_cache

    folder_ids = [root_folder_id]
    queue = [root_folder_id]

    while queue:
        current = queue.pop()
        results = service.files().list(
            q=f"'{current}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false",
            fields="files(id)"
        ).execute()
        for folder in results.get("files", []):
            folder_ids.append(folder["id"])
            queue.append(folder["id"])

    _folder_cache = folder_ids
    _cache_time = time.time()
    return _folder_cache


def search_files(service, query: str) -> list[dict]:
    folder_id = os.getenv("DRIVE_FOLDER_ID")
    if not folder_id:
        raise ValueError("DRIVE_FOLDER_ID not set")
    all_folder_ids = get_all_folder_ids(service, folder_id)

    parent_clause = " or ".join(f"'{fid}' in parents" for fid in all_folder_ids)
    q = f"({parent_clause}) and ({query}) and trashed = false"

    results = service.files().list(
        q=q,
        fields="files(id, name, mimeType, modifiedTime, webViewLink)"
    ).execute()

    return results.get("files", [])
