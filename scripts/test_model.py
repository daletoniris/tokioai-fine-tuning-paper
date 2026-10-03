#!/usr/bin/env python3
"""
Test a fine-tuned TokioAI model via Ollama.
Sends test queries and checks if the model generates correct tool calls.

Usage:
    python3 test_model.py tokioai-3b-v2
    python3 test_model.py tokioai-14b-v2
"""
import json
import re
import subprocess
import sys


# Test cases: (query, expected_tool_name)
TEST_CASES = [
    ("check disk space", "execute_local"),
    ("raspi temperature", "execute_raspi"),
    ("hay incidentes nuevos?", "irt_chat"),
    ("read my emails", "outlook_read"),
    ("show router interfaces", "execute_router"),
    ("check GCP VM status", "execute_gcp"),
    ("remember that the DB password changed", "memory"),
    ("scan ports on 10.0.0.1", "execute_local"),
]

SYSTEM_PROMPT = (
    "You are TokioAI -- specialized in cybersecurity, hacking, engineering, "
    "DevOps, and creative problem solving. "
    "Available tools: execute_local, execute_raspi, execute_gcp, execute_router, "
    "ssh_connect, read_file, write_file, edit_file, search_files, diagnose, "
    "mediator_exec, outlook_read, outlook_get, outlook_send, outlook_reply, "
    "teams_chats, teams_read, teams_send, wa_chats, wa_read, wa_send, "
    "irt_chat, irt_tool, irt_list_tools, memory, task. "
    "Be DIRECT. Act first, explain after."
)


def query_ollama(model: str, prompt: str) -> str:
    """Send a query to Ollama and return the response."""
    full_prompt = f"[SYSTEM]{SYSTEM_PROMPT}[/SYSTEM]\n[USER]{prompt}[/USER]\n[ASSISTANT]"
    result = subprocess.run(
        ["ollama", "run", model, full_prompt],
        capture_output=True, text=True, timeout=60,
    )
    return result.stdout.strip()


def extract_tool_name(response: str) -> str | None:
    """Extract the tool name from a <tool_call> block."""
    match = re.search(r'"name"\s*:\s*"(\w+)"', response)
    return match.group(1) if match else None


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "tokioai-3b-v2"
    print(f"Testing model: {model}")
    print("=" * 60)

    passed = 0
    total = len(TEST_CASES)

    for query, expected_tool in TEST_CASES:
        response = query_ollama(model, query)
        actual_tool = extract_tool_name(response)
        ok = actual_tool == expected_tool

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] '{query}'")
        print(f"       Expected: {expected_tool}")
        print(f"       Got:      {actual_tool}")
        if not ok:
            print(f"       Response: {response[:200]}")
        print()

        if ok:
            passed += 1

    print("=" * 60)
    print(f"Results: {passed}/{total} passed ({100 * passed / total:.0f}%)")

    if passed == total:
        print("ALL TESTS PASSED")
    else:
        print(f"FAILED: {total - passed} test(s)")
        sys.exit(1)


if __name__ == "__main__":
    main()
