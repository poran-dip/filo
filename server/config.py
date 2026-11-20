import json
import os

from dotenv import load_dotenv

load_dotenv()

_groq_api_key = os.getenv("GROQ_API_KEY")
_drive_folder_id = os.getenv("DRIVE_FOLDER_ID")
_google_service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

if not _groq_api_key:
    raise ValueError("GROQ_API_KEY is not set")
if not _drive_folder_id:
    raise ValueError("DRIVE_FOLDER_ID is not set")
if not _google_service_account_json:
    raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON is not set")

GROQ_API_KEY: str = _groq_api_key
DRIVE_FOLDER_ID: str = _drive_folder_id

try:
    SERVICE_ACCOUNT_INFO = json.loads(_google_service_account_json)
except json.JSONDecodeError as e:
    raise ValueError(f"GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON: {e}")
