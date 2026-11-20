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
            page = {
                "files": [{
                    "id": "a", "name": "notes", "mimeType": "text/plain",
                    "size": "10", "modifiedTime": "2024-01-01T00:00:00Z", "webViewLink": "https://x",
                }],
                "nextPageToken": "next",
            }
            api.list.return_value.execute.return_value = page
            message = tools.drive_list.invoke({
                "type": "tool_call", "id": "call", "name": "drive_list", "args": {}
            })
            # The model sees what it needs to navigate and compare (no webViewLink)...
            self.assertEqual(json.loads(message.content), {
                "files": [{
                    "id": "a", "name": "notes", "mimeType": "text/plain",
                    "modifiedTime": "2024-01-01T00:00:00Z", "size": "10",
                }],
                "nextPageToken": "next",
            })
            # ...while the UI artifact keeps the full metadata.
            self.assertEqual(message.artifact["files"], page["files"])
            self.assertFalse(message.artifact["continuation"])
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

    def test_model_can_compare_files_by_date_and_size(self):
        files = [
            {"id": "a", "name": "old", "mimeType": "text/plain", "size": "5",
             "modifiedTime": "2023-01-01T00:00:00Z", "webViewLink": "https://x/a"},
            {"id": "b", "name": "doc", "mimeType": "application/vnd.google-apps.document",
             "modifiedTime": "2024-06-01T00:00:00Z", "webViewLink": "https://x/b"},
        ]
        with patch.object(tools, "_service") as service:
            service.files.return_value.list.return_value.execute.return_value = {"files": files}
            message = tools.drive_list.invoke({
                "type": "tool_call", "id": "call", "name": "drive_list", "args": {}
            })
        seen = json.loads(message.content)["files"]
        self.assertEqual([f["modifiedTime"] for f in seen], ["2023-01-01T00:00:00Z", "2024-06-01T00:00:00Z"])
        self.assertEqual(seen[0]["size"], "5")
        self.assertNotIn("size", seen[1])  # absent for Google-native files, not null
        self.assertNotIn("webViewLink", message.content)

    def test_last_page_omits_next_page_token(self):
        with patch.object(tools, "_service") as service:
            service.files.return_value.list.return_value.execute.return_value = {"files": []}
            message = tools.drive_list.invoke({
                "type": "tool_call", "id": "call", "name": "drive_list", "args": {}
            })
            self.assertEqual(json.loads(message.content), {"files": []})


FOLDER = "application/vnd.google-apps.folder"


def _folder(fid, parent=None, **extra):
    item = {"id": fid, "mimeType": FOLDER, **extra}
    if parent is not None:
        item["parents"] = [parent]
    return item


def _http_error(status):
    return HttpError(Response({"status": status}), json.dumps({"error": {"message": "boom"}}).encode())


class DriveListScopeTests(unittest.TestCase):
    def _run(self, args, tree=None, get_error=None, list_error=None):
        """Invoke drive_list against a fake Drive; returns (reply, api mock)."""
        with patch.object(tools, "_service") as service:
            api = service.files.return_value
            if get_error:
                api.get.return_value.execute.side_effect = get_error
            else:
                api.get.side_effect = lambda fileId, fields: Mock(
                    execute=Mock(return_value=tree[fileId])
                )
            if list_error:
                api.list.return_value.execute.side_effect = list_error
            else:
                api.list.return_value.execute.return_value = {"files": []}
            message = tools.drive_list.invoke({
                "type": "tool_call", "id": "call", "name": "drive_list", "args": args
            })
            return message.content, api

    def test_rejects_malformed_folder_ids(self):
        for folder_id in ("x' or '1'='1", "a b", "a/b", "", "id)--", "x\n"):
            with self.subTest(folder_id=folder_id):
                reply, api = self._run({"folder_id": folder_id}, tree={})
                self.assertIn("inside the configured root", reply)
                api.get.assert_not_called()
                api.list.assert_not_called()

    def test_root_alias_is_denied(self):
        # Drive resolves "root" to the user's own root, whose id differs from "root".
        reply, api = self._run({"folder_id": "root_alias"}, tree={"root_alias": _folder("real-id", "elsewhere")})
        self.assertIn("inside the configured root", reply)
        api.list.assert_not_called()

    def test_multi_level_descendant_is_allowed(self):
        tree = {"c": _folder("c", "b"), "b": _folder("b", "a"), "a": _folder("a", "root")}
        _, api = self._run({"folder_id": "c"}, tree=tree)
        self.assertEqual(api.get.call_count, 3)
        self.assertIn("'c' in parents", api.list.call_args.kwargs["q"])

    def test_multi_level_chain_ending_outside_root_is_denied(self):
        tree = {"c": _folder("c", "b"), "b": _folder("b", "a"), "a": _folder("a")}
        reply, api = self._run({"folder_id": "c"}, tree=tree)
        self.assertIn("inside the configured root", reply)
        api.list.assert_not_called()

    def test_denies_trashed_folder_or_trashed_ancestor(self):
        trees = {
            "trashed folder": {"c": _folder("c", "root", trashed=True)},
            "trashed ancestor": {"c": _folder("c", "b"), "b": _folder("b", "root", trashed=True)},
        }
        for name, tree in trees.items():
            with self.subTest(name):
                reply, api = self._run({"folder_id": "c"}, tree=tree)
                self.assertIn("inside the configured root", reply)
                api.list.assert_not_called()

    def test_denies_non_folder_items(self):
        tree = {"f": {"id": "f", "mimeType": "application/pdf", "parents": ["root"]}}
        reply, api = self._run({"folder_id": "f"}, tree=tree)
        self.assertIn("inside the configured root", reply)
        api.list.assert_not_called()

    def test_denies_cyclic_ancestry(self):
        tree = {"a": _folder("a", "b"), "b": _folder("b", "a")}
        reply, api = self._run({"folder_id": "a"}, tree=tree)
        self.assertIn("inside the configured root", reply)
        api.list.assert_not_called()

    def test_ancestry_is_checked_on_continuation_pages(self):
        tree = {"c": _folder("c", "root")}
        _, api = self._run({"folder_id": "c", "page_token": "t"}, tree=tree)
        api.get.assert_called_once()
        self.assertEqual(api.list.call_args.kwargs["pageToken"], "t")

    def test_kind_filters(self):
        for kind, expected in (("folders", f"mimeType = '{FOLDER}'"), ("files", f"mimeType != '{FOLDER}'")):
            with self.subTest(kind=kind):
                _, api = self._run({"kind": kind})
                self.assertIn(expected, api.list.call_args.kwargs["q"])
        _, api = self._run({"kind": "all"})
        self.assertNotIn("mimeType", api.list.call_args.kwargs["q"])

    def test_drive_errors(self):
        for status, expected in ((403, "inside the configured root"), (404, "inside the configured root"),
                                 (400, "Retry from the first page")):
            with self.subTest(status=status, stage="get"):
                reply, api = self._run({"folder_id": "c"}, get_error=_http_error(status))
                self.assertIn(expected, reply)
                api.list.assert_not_called()
            with self.subTest(status=status, stage="list"):
                reply, _ = self._run({}, list_error=_http_error(status))
                self.assertIn(expected, reply)
        for stage in ("get", "list"):
            with self.subTest(status=500, stage=stage), self.assertRaises(HttpError):
                if stage == "get":
                    self._run({"folder_id": "c"}, get_error=_http_error(500))
                else:
                    self._run({}, list_error=_http_error(500))
