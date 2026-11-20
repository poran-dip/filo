import json
import os
import unittest
from unittest.mock import Mock, patch

from googleapiclient.errors import HttpError
from httplib2 import Response

# Supply inert settings and replace the import-time credentialed client.
with patch.dict(os.environ, GROQ_API_KEY="test", DRIVE_FOLDER_ID="root", GOOGLE_SERVICE_ACCOUNT_JSON="{}"):
    from server import drive
    with patch.object(drive, "get_drive_service"):
        from server import tools

from server.query import InvalidDriveQuery


class DriveSearchTests(unittest.TestCase):
    def test_invalid_filter_never_contacts_drive(self):
        service = Mock()
        with self.assertRaises(InvalidDriveQuery):
            drive.search_files(service, "invalid")
        self.assertEqual(service.mock_calls, [])

    def test_tool_returns_validation_feedback(self):
        reply = tools.drive_search.invoke({"query": "mimetype = 'x'"})
        self.assertIn("'mimetype'", reply)
        self.assertIn("mimeType", reply)

    def test_tool_handles_bad_request_and_propagates_other_errors(self):
        for status in (400, 403):
            error = HttpError(Response({"status": status}), json.dumps({
                "error": {"message": "invalid timestamp" if status == 400 else "forbidden"}
            }).encode())
            with self.subTest(status=status), patch.object(tools, "search_files", side_effect=error):
                args = {"query": "modifiedTime > 'yesterday'"}
                if status == 400:
                    self.assertIn("invalid timestamp", tools.drive_search.invoke(args))
                else:
                    with self.assertRaises(HttpError):
                        tools.drive_search.invoke(args)


class DriveListTests(unittest.TestCase):
    def test_lists_root_and_returns_page_for_display(self):
        with patch.object(tools, "_service") as service:
            api = service.files.return_value
            page = {"files": [{"id": "a", "name": "notes"}], "nextPageToken": "next"}
            api.list.return_value.execute.return_value = page
            message = tools.drive_list.invoke({
                "type": "tool_call", "id": "call", "name": "drive_list", "args": {}
            })
            self.assertEqual(json.loads(message.content), page)
            self.assertEqual(message.artifact["files"], page["files"])
            api.get.assert_not_called()
            self.assertEqual(api.list.call_args.kwargs["q"], "'root' in parents and trashed = false")

    def test_scoped_folder_and_page(self):
        with patch.object(tools, "_service") as service:
            api = service.files.return_value
            api.get.return_value.execute.return_value = {
                "id": "child", "mimeType": "application/vnd.google-apps.folder", "parents": ["root"]
            }
            api.list.return_value.execute.return_value = {"files": []}
            tools.drive_list.invoke({"folder_id": "child", "kind": "folders", "page_token": "next"})
            params = api.list.call_args.kwargs
            self.assertEqual(params["pageToken"], "next")
            self.assertIn("mimeType = 'application/vnd.google-apps.folder'", params["q"])

    def test_denies_unrelated_folder(self):
        with patch.object(tools, "_service") as service:
            api = service.files.return_value
            api.get.return_value.execute.return_value = {
                "id": "outside", "mimeType": "application/vnd.google-apps.folder"
            }
            tools.drive_list.invoke({"folder_id": "outside", "kind": "files"})
            api.list.assert_not_called()
