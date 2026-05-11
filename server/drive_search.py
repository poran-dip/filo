import os

def search_files(service, query: str):
    folder_id = os.getenv("DRIVE_FOLDER_ID")

    q = f"'{folder_id}' in parents and ({query})"

    results = service.files().list(
        q=q,
        fields="files(id, name, mimeType, modifiedTime, webViewLink)"
    ).execute()

    return results.get("files", [])
