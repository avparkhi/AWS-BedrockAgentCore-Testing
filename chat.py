"""
Interactive Chat Interface for AgentCore MicroVM Testing
========================================================
Commands:
  /new                  - Start a new session (new microVM, cold start)
  /session              - Show current session ID
  /sessions             - List all sessions used this run
  /switch <id>          - Switch to an existing session ID
  /ping                 - Ping current session (no LLM, shows microVM status)
  /user <1-5>           - Switch user (1=Maira, 2=Mani, 3=Mark, 4=Ishan, 5=Dhawal)
  /latency              - Show latency history for all requests
  /clear                - Clear screen
  /help                 - Show this help
  /quit                 - Exit

Just type normally to chat with the agent (LLM mode).
"""

import uuid
import json
import time
import os
import sys
from bedrock_agentcore_starter_toolkit import Runtime
from boto3.session import Session

# Colors
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
MAGENTA = "\033[35m"


def get_runtime() -> Runtime:
    with open("deploy_info.json") as f:
        info = json.load(f)
    runtime = Runtime()
    runtime.configure(
        entrypoint="strands_claude_context.py",
        auto_create_execution_role=True,
        auto_create_ecr=True,
        requirements_file="requirements.txt",
        region=info["region"],
        agent_name="strands_claude_context"
    )
    return runtime


def parse_ping_response(response) -> dict | None:
    """Extract ping JSON from response (dict or string)."""
    # If dict, look in 'response' key
    if isinstance(response, dict) and "response" in response:
        parts = response["response"]
        text = parts[0] if isinstance(parts, list) else str(parts)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass
    # Fallback: search in string
    response_str = str(response)
    try:
        if '"pong"' in response_str:
            start = response_str.index('{"pong"')
            depth = 0
            for i in range(start, len(response_str)):
                if response_str[i] == '{':
                    depth += 1
                elif response_str[i] == '}':
                    depth -= 1
                    if depth == 0:
                        return json.loads(response_str[start:i + 1])
    except (ValueError, json.JSONDecodeError):
        pass
    return None


def extract_response_text(response) -> str:
    """Extract the actual agent response text from the SDK response."""
    # If it's already a dict with 'response' key
    if isinstance(response, dict) and "response" in response:
        parts = response["response"]
        if isinstance(parts, list):
            return "\n".join(str(p) for p in parts)
        return str(parts)
    # If stringified
    import ast
    try:
        resp_dict = ast.literal_eval(str(response))
        if isinstance(resp_dict, dict) and "response" in resp_dict:
            parts = resp_dict["response"]
            if isinstance(parts, list):
                return "\n".join(str(p) for p in parts)
            return str(parts)
    except (ValueError, SyntaxError):
        pass
    return str(response)[:500]


def invoke(runtime, payload, session_id):
    start = time.perf_counter()
    response = runtime.invoke(payload, session_id=session_id)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, response


def print_header():
    print(f"\n{BOLD}{'=' * 65}{RESET}")
    print(f"{BOLD}  AgentCore MicroVM — Interactive Chat Interface{RESET}")
    print(f"{BOLD}{'=' * 65}{RESET}")
    print(f"  {DIM}Type /help for commands, or just type to chat{RESET}\n")


def print_status_bar(session_id, user_id, user_name, request_count):
    print(f"{DIM}┌─────────────────────────────────────────────────────────────────┐{RESET}")
    print(f"{DIM}│{RESET} {CYAN}Session:{RESET} {session_id}")
    print(f"{DIM}│{RESET} {CYAN}User:{RESET} {user_name} (id={user_id})  {CYAN}Requests:{RESET} {request_count}")
    print(f"{DIM}└─────────────────────────────────────────────────────────────────┘{RESET}")


def main():
    print_header()
    print(f"  {DIM}Connecting to AgentCore Runtime...{RESET}")
    runtime = get_runtime()
    print(f"  {GREEN}Connected!{RESET}\n")

    session_id = str(uuid.uuid4())
    user_id = "1"
    users = {"1": "Maira", "2": "Mani", "3": "Mark", "4": "Ishan", "5": "Dhawal"}
    request_count = 0
    latency_log = []
    sessions = {session_id: {"created": time.strftime("%H:%M:%S"), "requests": 0, "label": "initial"}}

    print_status_bar(session_id, user_id, users[user_id], request_count)

    while True:
        try:
            user_input = input(f"\n{GREEN}{users[user_id]}>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye!{RESET}")
            break

        if not user_input:
            continue

        # ── Commands ──
        if user_input == "/quit" or user_input == "/exit":
            print(f"{DIM}Goodbye!{RESET}")
            break

        elif user_input == "/help":
            print(__doc__)
            continue

        elif user_input == "/new":
            session_id = str(uuid.uuid4())
            sessions[session_id] = {"created": time.strftime("%H:%M:%S"), "requests": 0, "label": f"session-{len(sessions)+1}"}
            print(f"\n  {YELLOW}New session created (will trigger cold start){RESET}")
            print_status_bar(session_id, user_id, users[user_id], 0)
            continue

        elif user_input == "/session":
            print(f"\n  {CYAN}Current session:{RESET} {session_id}")
            print(f"  {CYAN}Requests on this session:{RESET} {sessions[session_id]['requests']}")
            continue

        elif user_input == "/sessions":
            print(f"\n  {CYAN}All sessions this run:{RESET}")
            for sid, info in sessions.items():
                marker = " <<< current" if sid == session_id else ""
                print(f"  {DIM}•{RESET} {sid}  ({info['created']}, {info['requests']} reqs){GREEN}{marker}{RESET}")
            continue

        elif user_input.startswith("/switch "):
            target = user_input[8:].strip()
            if target in sessions:
                session_id = target
                print(f"\n  {YELLOW}Switched to session:{RESET} {session_id}")
                print_status_bar(session_id, user_id, users[user_id], sessions[session_id]['requests'])
            else:
                # Try partial match
                matches = [s for s in sessions if s.startswith(target)]
                if len(matches) == 1:
                    session_id = matches[0]
                    print(f"\n  {YELLOW}Switched to session:{RESET} {session_id}")
                    print_status_bar(session_id, user_id, users[user_id], sessions[session_id]['requests'])
                else:
                    print(f"  {RED}Session not found. Use /sessions to list.{RESET}")
            continue

        elif user_input.startswith("/user "):
            new_uid = user_input[6:].strip()
            if new_uid in users:
                user_id = new_uid
                print(f"\n  {YELLOW}Switched to user:{RESET} {users[user_id]} (id={user_id})")
            else:
                print(f"  {RED}Invalid user. Choose 1-5: {', '.join(f'{k}={v}' for k,v in users.items())}{RESET}")
            continue

        elif user_input == "/ping":
            print(f"\n  {DIM}Pinging session {session_id}...{RESET}")
            try:
                latency_ms, response = invoke(runtime, {"mode": "ping"}, session_id)
                ping = parse_ping_response(response)
                request_count += 1
                sessions[session_id]['requests'] += 1

                if ping:
                    cache = ping.get("cache_status", "?")
                    cache_color = RED if cache == "COLD_START" else GREEN
                    print(f"\n  {BOLD}PONG{RESET}")
                    print(f"  ├─ {CYAN}Latency:{RESET}      {latency_ms:.0f} ms")
                    print(f"  ├─ {CYAN}Cache:{RESET}        {cache_color}{cache}{RESET}")
                    print(f"  ├─ {CYAN}Server time:{RESET}  {ping.get('elapsed_ms', '?')} ms")
                    print(f"  ├─ {CYAN}Invocations:{RESET} {ping.get('invocation_count', '?')}")
                    print(f"  ├─ {CYAN}Init time:{RESET}    {ping.get('init_time', '?')}")
                    print(f"  └─ {CYAN}Session:{RESET}      {ping.get('session_id', '?')}")
                else:
                    print(f"  {YELLOW}Response:{RESET} {response[:200]}")
                    print(f"  {CYAN}Latency:{RESET} {latency_ms:.0f} ms")

                latency_log.append({"type": "ping", "session": session_id[:12], "latency_ms": round(latency_ms), "cache": ping.get("cache_status", "?") if ping else "?"})
            except Exception as e:
                print(f"  {RED}Error:{RESET} {e}")
            continue

        elif user_input == "/latency":
            if not latency_log:
                print(f"  {DIM}No requests yet.{RESET}")
                continue
            print(f"\n  {CYAN}{'#':<4} {'Type':<8} {'Session':<14} {'Latency':>10} {'Cache':<12}{RESET}")
            print(f"  {DIM}{'─' * 52}{RESET}")
            for i, entry in enumerate(latency_log):
                cache_color = RED if entry.get('cache') == 'COLD_START' else GREEN if entry.get('cache') == 'WARM' else ""
                print(f"  {i+1:<4} {entry['type']:<8} {entry['session']:<14} {entry['latency_ms']:>8} ms {cache_color}{entry.get('cache', '')}{RESET}")
            continue

        elif user_input == "/clear":
            os.system("clear")
            print_header()
            print_status_bar(session_id, user_id, users[user_id], sessions[session_id]['requests'])
            continue

        elif user_input.startswith("/"):
            print(f"  {RED}Unknown command. Type /help{RESET}")
            continue

        # ── Chat mode (LLM call) ──
        print(f"\n  {DIM}Sending to session {session_id}...{RESET}")
        try:
            latency_ms, response = invoke(
                runtime,
                {"prompt": user_input, "user_id": user_id},
                session_id
            )
            request_count += 1
            sessions[session_id]['requests'] += 1

            response_text = extract_response_text(response)

            print(f"\n  {MAGENTA}Agent:{RESET} {response_text}")
            print(f"\n  {DIM}Session: {session_id}{RESET}")
            print(f"  {DIM}Latency: {latency_ms:.0f} ms | Requests on session: {sessions[session_id]['requests']}{RESET}")

            latency_log.append({"type": "chat", "session": session_id[:12], "latency_ms": round(latency_ms), "cache": ""})

        except Exception as e:
            print(f"  {RED}Error:{RESET} {e}")


if __name__ == "__main__":
    main()
