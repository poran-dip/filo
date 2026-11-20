import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
run_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_eval)


def _result(name="drive_search", content="{}", status="success"):
    return SimpleNamespace(name=name, content=content, status=status)


CASE = {"expected_ids": ["a", "b"], "rounds": [[{"name": "drive_search", "args": {}}]]}


class GradeTests(unittest.TestCase):
    def test_exact_match_passes(self):
        self.assertEqual(run_eval.grade(CASE, ["b", "a"], [_result()], live=True), (True, []))

    def test_extra_missing_and_duplicate_cards_fail(self):
        for ids in (["a", "b", "c"], ["a"], ["a", "a", "b"]):
            with self.subTest(ids=ids):
                ok, problems = run_eval.grade(CASE, ids, [_result()], live=False)
                self.assertFalse(ok)
                self.assertIn("cards", problems[0])

    def test_wrong_tool_fails_live_but_replay_cannot_judge_tools(self):
        trace = [_result(name="drive_list")]
        self.assertFalse(run_eval.grade(CASE, ["a", "b"], trace, live=True)[0])
        self.assertTrue(run_eval.grade(CASE, ["a", "b"], trace, live=False)[0])

    def test_invalid_tool_results_fail_even_with_right_cards(self):
        for bad in (_result(content="Invalid search filter: nope."), _result(status="error")):
            with self.subTest(content=bad.content, status=bad.status):
                ok, problems = run_eval.grade(CASE, ["a", "b"], [bad], live=False)
                self.assertFalse(ok)
                self.assertIn("invalid", problems[0])

    def test_zero_hit_case_needs_a_real_search(self):
        case = {"expected_ids": [], "rounds": [[{"name": "drive_search", "args": {}}]]}
        self.assertTrue(run_eval.grade(case, [], [_result(content="No files found.")], live=True)[0])
        self.assertFalse(run_eval.grade(case, [], [_result(content="Invalid search filter: x")], live=True)[0])

    def test_no_tool_case_fails_if_a_tool_is_used(self):
        case = {"expected_ids": [], "rounds": []}
        self.assertTrue(run_eval.grade(case, [], [], live=True)[0])
        self.assertFalse(run_eval.grade(case, [], [_result(content="No files found.")], live=True)[0])


class ExitCodeTests(unittest.TestCase):
    def stats(self, **overrides):
        return {"runs": 10, "passed": 10, "errors": 0, "expected_total": 20, **overrides}

    def test_replay_fails_on_any_exact_case_failure(self):
        self.assertEqual(run_eval.exit_code(self.stats(), live=False, min_pass=None), 0)
        self.assertEqual(run_eval.exit_code(self.stats(passed=9), live=False, min_pass=None), 1)

    def test_live_is_report_only_unless_min_pass_is_given(self):
        low = self.stats(passed=5)
        self.assertEqual(run_eval.exit_code(low, live=True, min_pass=None), 0)
        self.assertEqual(run_eval.exit_code(low, live=True, min_pass=0.9), 1)
        self.assertEqual(run_eval.exit_code(self.stats(passed=9), live=True, min_pass=0.9), 0)

    def test_errors_and_empty_expectations_always_fail(self):
        for live in (True, False):
            with self.subTest(live=live):
                self.assertEqual(run_eval.exit_code(self.stats(errors=1), live=live, min_pass=None), 1)
                self.assertEqual(run_eval.exit_code(self.stats(expected_total=0), live=live, min_pass=None), 1)


class AttemptTests(unittest.TestCase):
    def test_retries_then_succeeds_with_backoff(self):
        calls, sleeps, retried = [], [], []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("rate limited")
            return "ok"

        result = run_eval.attempt(flaky, retries=2, backoff=5, sleep=sleeps.append,
                                  on_retry=lambda n, exc: retried.append(n))
        self.assertEqual((result, sleeps, retried), ("ok", [5, 10], [1, 2]))

    def test_raises_after_retries_are_exhausted(self):
        calls = []

        def broken():
            calls.append(1)
            raise RuntimeError("down")

        with self.assertRaises(RuntimeError):
            run_eval.attempt(broken, retries=1, backoff=0, sleep=lambda _: None)
        self.assertEqual(len(calls), 2)

    def test_no_retries_means_one_try(self):
        calls = []

        def broken():
            calls.append(1)
            raise RuntimeError("down")

        with self.assertRaises(RuntimeError):
            run_eval.attempt(broken, retries=0, backoff=0, sleep=lambda _: None)
        self.assertEqual(len(calls), 1)


class RunnerTests(unittest.TestCase):
    def _run(self, *args):
        env = {k: v for k, v in os.environ.items() if k != "GROQ_API_KEY"}
        return subprocess.run([sys.executable, str(ROOT / "eval" / "run_eval.py"), *args],
                              capture_output=True, text=True, cwd=ROOT, env=env)

    def test_replay_of_the_committed_suite_passes_exactly(self):
        done = self._run("--replay")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("Scripted replay: no model is called", done.stdout)

    def test_live_mode_requires_a_key_instead_of_silently_replaying(self):
        done = self._run()
        self.assertEqual(done.returncode, 2)
        self.assertIn("GROQ_API_KEY", done.stderr)


if __name__ == "__main__":
    unittest.main()
