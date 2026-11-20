import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

# Supply inert settings; agent import builds the model and tools at import time.
with patch.dict(
    os.environ,
    GROQ_API_KEY="test",
    DRIVE_FOLDER_ID="root",
    GOOGLE_SERVICE_ACCOUNT_JSON="{}",
):
    from server import drive

    with patch.object(drive, "get_drive_service"):
        from server import agent


def _list(files, continuation=False, call_id="1"):
    return ToolMessage(
        content="{}",
        name="drive_list",
        tool_call_id=call_id,
        artifact={"files": files, "continuation": continuation},
    )


def _search(files, call_id="1"):
    return ToolMessage(
        content="{}", artifact=files, name="drive_search", tool_call_id=call_id
    )


def _search_round(*queries):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "drive_search", "id": call_id, "args": {"query": query}}
            for call_id, query in queries
        ],
    )


def _list_round(call_id="1"):
    return AIMessage(
        content="", tool_calls=[{"name": "drive_list", "id": call_id, "args": {}}]
    )


class RunAgentTests(unittest.TestCase):
    def _run(self, *tool_messages, history=None):
        reply = AIMessage(content="done")
        with patch.object(agent, "graph") as graph:
            graph.invoke.return_value = {"messages": [*tool_messages, reply]}
            result = agent.run_agent(history or [{"role": "user", "content": "hi"}])
        return result, graph

    def test_prompt_is_rebuilt_per_request_and_sent_first(self):
        _, graph = self._run(
            history=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
            ]
        )
        sent = graph.invoke.call_args.args[0]["messages"]
        self.assertIsInstance(sent[0], SystemMessage)
        self.assertIn("drive_list", sent[0].content)
        self.assertIsInstance(sent[1], HumanMessage)
        self.assertIsInstance(sent[2], AIMessage)

    def test_history_is_limited_to_last_six_messages(self):
        history = [{"role": "user", "content": str(i)} for i in range(10)]
        _, graph = self._run(history=history)
        sent = graph.invoke.call_args.args[0]["messages"]
        self.assertEqual([m.content for m in sent[1:]], ["4", "5", "6", "7", "8", "9"])

    def test_single_list_page(self):
        (reply, files), _ = self._run(_list([{"id": "a"}]))
        self.assertEqual(reply, "done")
        self.assertEqual(files, [{"id": "a"}])

    def test_continuation_pages_accumulate(self):
        (_, files), _ = self._run(
            _list([{"id": "a"}], call_id="1"),
            _list([{"id": "b"}], continuation=True, call_id="2"),
            _list([{"id": "c"}], continuation=True, call_id="3"),
        )
        self.assertEqual([f["id"] for f in files], ["a", "b", "c"])

    def test_new_listing_replaces_previous_one(self):
        (_, files), _ = self._run(
            _list([{"id": "a"}], call_id="1"),
            _list([{"id": "b"}], call_id="2"),
        )
        self.assertEqual(files, [{"id": "b"}])

    def test_search_after_list_replaces_files(self):
        (_, files), _ = self._run(
            _list([{"id": "a"}], call_id="1"), _search([{"id": "s"}], call_id="2")
        )
        self.assertEqual(files, [{"id": "s"}])

    def test_parallel_searches_merge_by_id(self):
        updated = {"id": "a", "name": "renamed", "webViewLink": "https://example.com/a"}
        (_, files), _ = self._run(
            _search_round(
                ("1", "name contains 'report'"), ("3", "name contains 'budget'")
            ),
            _search([{"id": "a"}, {"id": "b"}], call_id="1"),
            _search([{"id": "c"}, updated], call_id="3"),
        )
        self.assertEqual(files, [updated, {"id": "b"}, {"id": "c"}])

    def test_refinement_replaces_broad_results_even_when_empty(self):
        for refined in ([{"id": "b"}], []):
            with self.subTest(refined=refined):
                (_, files), _ = self._run(
                    _search_round(("1", "name contains 'report'")),
                    _search([{"id": "a"}, {"id": "b"}, {"id": "c"}], call_id="1"),
                    _search_round(("2", "name = 'annual report'")),
                    _search(refined, call_id="2"),
                )
                self.assertEqual(files, refined)

    def test_failed_parallel_search_does_not_clear_sibling_hits(self):
        invalid = ToolMessage(
            content="Invalid search filter: nope.",
            name="drive_search",
            tool_call_id="2",
            artifact=[],
        )
        (_, files), _ = self._run(
            _search_round(("1", "name contains 'a'"), ("2", "bad")),
            _search([{"id": "a"}], call_id="1"),
            invalid,
        )
        self.assertEqual(files, [{"id": "a"}])

    def test_search_after_navigation_starts_fresh(self):
        (_, files), _ = self._run(
            _search_round(("1", "name contains 'a'")),
            _search([{"id": "a"}, {"id": "b"}], call_id="1"),
            _list_round("2"),
            _list([{"id": "folder"}], call_id="2"),
            _search_round(("3", "name contains 'c'")),
            _search([{"id": "c"}], call_id="3"),
        )
        self.assertEqual(files, [{"id": "c"}])

    def test_denied_listing_after_search_keeps_search_hits(self):
        denied = ToolMessage(
            content="Choose a folder inside the configured root.",
            name="drive_list",
            tool_call_id="2",
            artifact={},
        )
        (_, files), _ = self._run(
            _search_round(("1", "name contains 'a'")),
            _search([{"id": "a"}], call_id="1"),
            _list_round("2"),
            denied,
        )
        self.assertEqual(files, [{"id": "a"}])

    def test_listing_continuation_after_search_shows_only_listing(self):
        (_, files), _ = self._run(
            _list([{"id": "a"}], call_id="1"),
            _search([{"id": "s"}], call_id="2"),
            _list([{"id": "b"}], continuation=True, call_id="3"),
        )
        self.assertEqual(files, [{"id": "a"}, {"id": "b"}])

    def test_empty_or_failed_search_clears_earlier_files(self):
        for content in (
            "No files found.",
            "Invalid search filter: nope. Revise the query and try again.",
        ):
            with self.subTest(content=content):
                empty = ToolMessage(
                    content=content, name="drive_search", tool_call_id="2", artifact=[]
                )
                (_, files), _ = self._run(_list([{"id": "a"}], call_id="1"), empty)
                self.assertEqual(files, [])

    def test_prompt_warns_against_superlatives_on_truncated_results(self):
        prompt = agent.build_system_prompt()
        self.assertIn('"truncated": true', prompt)
        self.assertIn("Never claim a newest", prompt)

    def test_empty_artifact_from_denied_listing_keeps_earlier_files(self):
        denied = ToolMessage(
            content="Choose a folder inside the configured root.",
            name="drive_list",
            tool_call_id="2",
            artifact={},
        )
        (_, files), _ = self._run(_list([{"id": "a"}], call_id="1"), denied)
        self.assertEqual(files, [{"id": "a"}])

    def test_no_tool_calls_returns_no_files(self):
        (_, files), _ = self._run()
        self.assertEqual(files, [])


if __name__ == "__main__":
    unittest.main()
