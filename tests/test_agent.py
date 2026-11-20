import os
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

# Supply inert settings; agent import builds the model and tools at import time.
with patch.dict(os.environ, GROQ_API_KEY="test", DRIVE_FOLDER_ID="root", GOOGLE_SERVICE_ACCOUNT_JSON="{}"):
    from server import drive
    with patch.object(drive, "get_drive_service"):
        from server import agent


def _list(files, continuation=False, call_id="1"):
    return ToolMessage(content="{}", name="drive_list", tool_call_id=call_id,
                       artifact={"files": files, "continuation": continuation})


def _search(files, call_id="1"):
    return ToolMessage(content="{}", artifact=files, name="drive_search", tool_call_id=call_id)


class RunAgentTests(unittest.TestCase):
    def _run(self, *tool_messages, history=None):
        reply = AIMessage(content="done")
        with patch.object(agent, "graph") as graph:
            graph.invoke.return_value = {"messages": [*tool_messages, reply]}
            result = agent.run_agent(history or [{"role": "user", "content": "hi"}])
        return result, graph

    def test_prompt_is_rebuilt_per_request_and_sent_first(self):
        _, graph = self._run(history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}])
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
        (_, files), _ = self._run(_list([{"id": "a"}], call_id="1"), _search([{"id": "s"}], call_id="2"))
        self.assertEqual(files, [{"id": "s"}])

    def test_empty_or_failed_search_clears_earlier_files(self):
        for content in ("No files found.", "Invalid search filter: nope. Revise the query and try again."):
            with self.subTest(content=content):
                empty = ToolMessage(content=content, name="drive_search", tool_call_id="2", artifact=[])
                (_, files), _ = self._run(_list([{"id": "a"}], call_id="1"), empty)
                self.assertEqual(files, [])

    def test_prompt_warns_against_superlatives_on_truncated_results(self):
        prompt = agent.build_system_prompt()
        self.assertIn('"truncated": true', prompt)
        self.assertIn("Never claim a newest", prompt)

    def test_empty_artifact_from_denied_listing_keeps_earlier_files(self):
        denied = ToolMessage(content="Choose a folder inside the configured root.",
                             name="drive_list", tool_call_id="2", artifact={})
        (_, files), _ = self._run(_list([{"id": "a"}], call_id="1"), denied)
        self.assertEqual(files, [{"id": "a"}])

    def test_no_tool_calls_returns_no_files(self):
        (_, files), _ = self._run()
        self.assertEqual(files, [])


if __name__ == "__main__":
    unittest.main()
