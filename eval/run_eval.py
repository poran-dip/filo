"""Evaluate Filo's Drive agent against a fixed, local Drive folder tree.

Default: the real Groq model chooses the tool calls (needs GROQ_API_KEY). Drive is still the local
fixture, so no Drive credentials are needed. The production graph, tools and card selection run unchanged.

--replay: no model is called. Each case's reference tool calls are replayed, which checks only the
harness, the tools and card selection. It is deterministic and is the one mode that can be gated exactly.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from contextlib import nullcontext
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from fixture_drive import FixtureDrive


def replay(case):
    for i, calls in enumerate(case["rounds"]):
        yield AIMessage(content="", tool_calls=[
            {**call, "id": f"call-{i}-{j}"} for j, call in enumerate(calls)
        ])
    yield AIMessage(content="done")


def turn(agent, case, history, live):
    graph = agent.graph
    responses = (ChatResult(generations=[ChatGeneration(message=msg)]) for msg in replay(case))
    model = nullcontext() if live else patch.object(type(agent.llm), "_generate", side_effect=responses)
    # Observe the production graph at run_agent's invocation boundary.
    with model, patch.object(agent, "graph", wraps=graph) as observed:
        traces = []

        def invoke(state):
            result = graph.invoke(state)
            traces.extend(m for m in result["messages"] if isinstance(m, ToolMessage))
            return result

        observed.invoke.side_effect = invoke
        reply, files = agent.run_agent(history + [{"role": "user", "content": case["message"]}])
    return reply, [f["id"] for f in files], traces


def grade(case, ids, trace, live):
    """Return (passed, problems) for one run of a case. Extra cards, missing cards and bad calls all fail."""
    expected = sorted(case["expected_ids"])
    problems = []
    if sorted(ids) != expected:
        problems.append(f"cards {sorted(ids)}, expected {expected}")
    # Replay's tool calls come from the case itself, so only a live model can pick the wrong tools.
    required = {call["name"] for calls in case["rounds"] for call in calls}
    used = {msg.name for msg in trace}
    if live and used != required:
        problems.append(f"tools {sorted(used)}, expected {sorted(required)}")
    invalid = {msg.name for msg in trace
               if msg.status == "error"
               or not (msg.content == "No files found." or msg.content.startswith(("{", "[")))}
    if invalid:
        problems.append(f"invalid tool results from {sorted(invalid)}")
    return not problems, problems


def attempt(run_turn, retries, backoff, sleep=time.sleep, on_retry=lambda n, exc: None):
    """Run a turn, retrying provider/network errors. Retries are for infrastructure, not for bad answers."""
    for n in range(retries + 1):
        try:
            return run_turn()
        except Exception as exc:
            if n == retries:
                raise
            on_retry(n + 1, exc)
            sleep(backoff * (n + 1))


def exit_code(stats, *, live, min_pass):
    if stats["errors"] or not stats["expected_total"]:
        return 1
    if not live:
        return int(stats["passed"] != stats["runs"])  # deterministic: any exact-case failure fails
    return int(min_pass is not None and stats["passed"] < min_pass * stats["runs"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay", action="store_true", help="replay reference tool calls instead of calling Groq")
    parser.add_argument("--runs", type=int, default=1, help="runs per case (live output varies between runs)")
    parser.add_argument("--retries", type=int, default=2, help="retries per turn on provider or network errors (live)")
    parser.add_argument("--backoff", type=float, default=10.0, help="seconds to wait before retry n, times n")
    parser.add_argument("--min-pass", type=float, default=None,
                        help="live only: fail if fewer than this fraction of case runs pass (default: report only)")
    args = parser.parse_args(argv)
    live = not args.replay
    key = os.environ.get("GROQ_API_KEY", "")
    if live and not key:
        parser.error("live mode needs GROQ_API_KEY (use --replay for the offline harness check)")
    if args.runs < 1 or args.retries < 0 or (args.min_pass is not None and not 0 <= args.min_pass <= 1):
        parser.error("--runs must be >= 1, --retries >= 0 and --min-pass between 0 and 1")

    cases = json.loads((HERE / "cases.json").read_text())
    if not cases:
        parser.error("empty eval suite")
    fixture = FixtureDrive(json.loads((HERE / "fixture.json").read_text()))
    settings = {"DRIVE_FOLDER_ID": "root", "GOOGLE_SERVICE_ACCOUNT_JSON": "{}"}
    if not live:
        settings["GROQ_API_KEY"] = "offline"
    with patch.dict(os.environ, settings):
        from server import drive
        with patch.object(drive, "get_drive_service", return_value=fixture):
            from server import agent

    mode = "live Groq" if live else "scripted replay"
    if not live:
        print("Scripted replay: no model is called. This checks the harness, tools and card selection only.")
    by_message = {case["message"]: case for case in cases}
    stats = dict(runs=0, passed=0, errors=0, retries=0, retrieved=0, expected_total=0, extra_cards=0)
    passes = {case["message"]: 0 for case in cases}

    def note_retry(n, exc):
        stats["retries"] += 1
        print(f"RETRY {n} after {type(exc).__name__}")

    def one_turn(case):
        history = []
        if previous := case.get("previous_message"):
            reply, _, _ = turn(agent, by_message[previous], [], live)
            history = [{"role": "user", "content": previous}, {"role": "assistant", "content": reply}]
        return turn(agent, case, history, live)

    for case in cases:
        expected_ids = set(case["expected_ids"])
        for _ in range(args.runs):
            stats["runs"] += 1
            stats["expected_total"] += len(expected_ids)
            try:
                _, ids, trace = attempt(lambda: one_turn(case), args.retries if live else 0,
                                        args.backoff, on_retry=note_retry)
            except Exception as exc:
                stats["errors"] += 1
                message = str(exc).replace(key, "***") if key else str(exc)
                print(f"ERROR {case['message']} | {type(exc).__name__}: {message[:300]}")
                continue
            stats["retrieved"] += len(expected_ids & set(ids))
            stats["extra_cards"] += len(set(ids) - expected_ids)
            ok, problems = grade(case, ids, trace, live)
            if ok:
                stats["passed"] += 1
                passes[case["message"]] += 1
            else:
                print(f"FAIL {case['message']} | " + "; ".join(problems))

    if args.runs > 1:
        for message, count in passes.items():
            if 0 < count < args.runs:
                print(f"FLAKY {message} | passed {count}/{args.runs}")
    runs = stats["runs"]
    print(f"{mode}: {stats['passed']}/{runs} case runs passed ({100 * stats['passed'] / runs:.1f}%)"
          f"{', errored: ' + str(stats['errors']) if stats['errors'] else ''}"
          f"{', provider retries: ' + str(stats['retries']) if stats['retries'] else ''}")
    if not stats["expected_total"]:
        print("Recall undefined: no expected file IDs.")
    else:
        print(f"{mode} recall: {stats['retrieved']}/{stats['expected_total']} "
              f"({100 * stats['retrieved'] / stats['expected_total']:.2f}%); "
              f"unexpected cards: {stats['extra_cards']}")
    if live:
        print("Gate: " + (f"pass rate >= {args.min_pass:.0%}" if args.min_pass is not None
                          else "none (report only; pass --min-pass to gate)"))
    else:
        print("Gate: every case must pass exactly")
    return exit_code(stats, live=live, min_pass=args.min_pass)


if __name__ == "__main__":
    sys.exit(main())
