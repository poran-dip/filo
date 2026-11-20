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
