"""Score the production graph against a fixed, local Drive folder tree."""
import argparse
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

from contextlib import nullcontext
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.messages import AIMessage, ToolMessage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="let Groq choose the tool calls")
    args = parser.parse_args()
    if args.live and not os.environ.get("GROQ_API_KEY"):
        parser.error("--live requires GROQ_API_KEY")
    cases = json.loads((HERE / "cases.json").read_text())
    if not cases:
        parser.error("empty eval suite")
    fixture = FixtureDrive(json.loads((HERE / "fixture.json").read_text()))
    settings = {"DRIVE_FOLDER_ID": "root", "GOOGLE_SERVICE_ACCOUNT_JSON": "{}"}
    if not args.live:
        settings["GROQ_API_KEY"] = "offline"
    with patch.dict(os.environ, settings):
        from server import drive
        with patch.object(drive, "get_drive_service", return_value=fixture):
            from server import agent
    by_message = {case["message"]: case for case in cases}
    passed = 0
    retrieved = 0
    expected_total = 0
    errors = 0
    for case in cases:
        expected_ids = set(case["expected_ids"])
        expected_total += len(expected_ids)
        try:
            history = []
            if previous := case.get("previous_message"):
                reply, _, _ = turn(agent, by_message[previous], [], args.live)
                history = [{"role": "user", "content": previous}, {"role": "assistant", "content": reply}]
            _, ids, trace = turn(agent, case, history, args.live)
            retrieved += len(expected_ids.intersection(ids))
            expected = sorted(case["expected_ids"])
            required = {call["name"] for calls in case["rounds"] for call in calls}
            used = {msg.name for msg in trace}
            # An invalid tool call must not pass a zero-hit case by returning no cards.
            valid = all(msg.status != "error" and
                        (msg.content == "No files found." or msg.content.startswith(("{", "["))) for msg in trace)
            if sorted(ids) == expected and required == used and valid:
                passed += 1
            else:
                print(f"FAIL {case['message']} | expected {expected}, got {ids}; "
                      f"tools {sorted(used)}, expected {sorted(required)}; valid calls: {valid}")
        except Exception as exc:
            errors += 1
            print(f"ERROR {case['message']} | {type(exc).__name__}")
    mode = "live Groq" if args.live else "scripted replay"
    print(f"{mode}: {passed}/{len(cases)} passed ({100 * passed / len(cases):.1f}%)")
    if not expected_total:
        print("Recall undefined: no expected file IDs.")
        return 1
    print(f"{mode} micro recall: {retrieved}/{expected_total} "
          f"({100 * retrieved / expected_total:.2f}%), required >90%")
    return int(errors > 0 or 100 * retrieved <= 90 * expected_total)


if __name__ == "__main__":
    sys.exit(main())
