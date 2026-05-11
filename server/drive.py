import os
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build

load_dotenv()
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        scopes=SCOPES
    )

    return build("drive", "v3", credentials=creds)


def search_files(service, query: str):
    folder_id = os.getenv("DRIVE_FOLDER_ID")

    q = f"'{folder_id}' in parents and ({query})"

    results = service.files().list(
        q=q,
        fields="files(id, name, mimeType, modifiedTime, webViewLink)"
    ).execute()

    return results.get("files", [])
