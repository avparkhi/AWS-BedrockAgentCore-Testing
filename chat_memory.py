"""
Interactive Chat Interface for Memory Agent
============================================
Commands:
  /new                  - Start a new runtime session (new microVM, cold start)
  /session              - Show current runtime session ID
  /sessions             - List all runtime sessions used this run
  /switch <id>          - Switch to an existing runtime session ID
  /ping                 - Ping current session (no LLM, shows microVM status)
  /user <1-5>           - Switch user (1=Maira, 2=Mani, 3=Mark, 4=Ishan, 5=Dhawal)
  /memory               - Show last 5 turns from durable memory
  /forget               - Clear durable memory (switches to new memory session)
  /msession             - Show current memory session ID
  /msession <id>        - Change memory session ID
  /latency              - Show latency history for all requests
  /clear                - Clear screen
  /help                 - Show this help
  /quit                 - Exit

Just type normally to chat with the agent (hybrid memory mode).

Memory Architecture:
  - RAM history: Fast, same microVM session only (resets on /new)
  - Durable memory: Persists across sessions, restarts, redeploys
  - On warm starts: RAM used for context (fast, no API call)
  - On cold starts: Durable memory loaded automatically
"""

import uuid
import json
import time
import os
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
BLUE = "\033[34m"


def get_runtime() -> Runtime:
    with open("deploy_memory_info.json") as f:
        info = json.load(f)
    runtime = Runtime()
    runtime.configure(
        entrypoint="memory_agent.py",
        auto_create_execution_role=True,
        auto_create_ecr=True,
        requirements_file="requirements.txt",
        region=info["region"],
        agent_name="memory_agent",
    )
    return runtime, info


def parse_ping_response(response) -> dict | None:
    """Extract ping JSON from response (dict or string)."""
    if isinstance(response, dict) and "response" in response:
        parts = response["response"]
        text = parts[0] if isinstance(parts, list) else str(parts)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass
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


def extract_response_text(response) -> tuple[str, str, str, str]:
    """Extract response text and memory metadata.

    Returns (text, memory_source, cache_status, agent_session_id).

    The agent returns: <<META:key=val,key=val>>\\nresponse_text
    The SDK wraps this in {"response": [text]} or similar.
    """
    import re

    # Step 1: Unwrap SDK envelope to get the raw agent output
    raw = ""
    if isinstance(response, dict) and "response" in response:
        parts = response["response"]
        if isinstance(parts, list):
            raw = "\n".join(str(p) for p in parts)
        else:
            raw = str(parts)
    else:
        raw = str(response)

    # Step 1b: SDK may JSON-encode the string (escaping \n, unicode, etc.)
    # Try to decode it — could be wrapped in quotes or just have escaped chars
    if raw.startswith('"') and raw.endswith('"'):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
    # Also try decoding as a JSON string value (handles \n, \t, \uXXXX)
    try:
        decoded = json.loads('"' + raw.replace('\\', '\\\\').replace('"', '\\"') + '"')
        raw = decoded
    except (json.JSONDecodeError, TypeError):
        pass
    # Fallback: manually unescape common sequences
    if '\\n' in raw:
        raw = raw.replace('\\n', '\n').replace('\\t', '\t')
    # Strip surrounding quotes if present
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1]

    # Step 2: Parse our <<META:...>> header
    memory_source = "?"
    cache_status = "?"
    agent_session_id = "?"

    meta_match = re.search(r"<<META:(.*?)>>", raw)
    if meta_match:
        meta_str = meta_match.group(1)
        for pair in meta_str.split(","):
            if "=" in pair:
                key, val = pair.split("=", 1)
                if key == "memory_source":
                    memory_source = val
                elif key == "cache_status":
                    cache_status = val
                elif key == "session_id":
                    agent_session_id = val
        # Remove the META line from response text
        text = re.sub(r"<<META:.*?>>\n?", "", raw).strip()
    else:
        text = raw.strip()

    return text, memory_source, cache_status, agent_session_id


def invoke(runtime, payload, session_id):
    start = time.perf_counter()
    response = runtime.invoke(payload, session_id=session_id)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, response


def print_header():
    print(f"\n{BOLD}{'=' * 65}{RESET}")
    print(f"{BOLD}  AgentCore MicroVM — Memory Agent Chat Interface{RESET}")
    print(f"{BOLD}{'=' * 65}{RESET}")
    print(f"  {DIM}Type /help for commands, or just type to chat{RESET}")
    print(f"  {DIM}Hybrid memory: RAM (same session) + Durable (cross-session){RESET}\n")


def print_status_bar(session_id, memory_session_id, user_id, user_name, request_count):
    print(f"{DIM}┌─────────────────────────────────────────────────────────────────┐{RESET}")
    print(f"{DIM}│{RESET} {CYAN}Runtime Session:{RESET} {session_id}")
    print(f"{DIM}│{RESET} {BLUE}Memory Session:{RESET}  {memory_session_id}")
    print(f"{DIM}│{RESET} {CYAN}User:{RESET} {user_name} (id={user_id})  {CYAN}Requests:{RESET} {request_count}")
    print(f"{DIM}└─────────────────────────────────────────────────────────────────┘{RESET}")


def memory_source_label(source):
    """Return colored label for memory source."""
    labels = {
        "ram": f"{GREEN}RAM{RESET}",
        "durable": f"{BLUE}DURABLE{RESET}",
        "both": f"{MAGENTA}RAM+DURABLE{RESET}",
        "none": f"{DIM}none{RESET}",
    }
    return labels.get(source, f"{DIM}{source}{RESET}")


def main():
    print_header()
    print(f"  {DIM}Connecting to AgentCore Runtime...{RESET}")
    runtime, deploy_info = get_runtime()
    memory_id = deploy_info.get("memory_id")
    print(f"  {GREEN}Connected!{RESET}")
    if memory_id:
        print(f"  {BLUE}Memory ID:{RESET} {memory_id}\n")
    else:
        print(f"  {YELLOW}Warning: No memory_id in deploy_memory_info.json{RESET}\n")

    session_id = str(uuid.uuid4())
    memory_session_id = "default"
    last_agent_session_id = "?"
    user_id = "1"
    users = {"1": "Maira", "2": "Mani", "3": "Mark", "4": "Ishan", "5": "Dhawal"}
    request_count = 0
    latency_log = []
    sessions = {session_id: {"created": time.strftime("%H:%M:%S"), "requests": 0, "label": "initial"}}

    print_status_bar(session_id, memory_session_id, user_id, users[user_id], request_count)

    while True:
        try:
            user_input = input(f"\n{GREEN}{users[user_id]}>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye!{RESET}")
            break

        if not user_input:
            continue

        # ── Commands ──
        if user_input in ("/quit", "/exit"):
            print(f"{DIM}Goodbye!{RESET}")
            break

        elif user_input == "/help":
            print(__doc__)
            continue

        elif user_input == "/new":
            session_id = str(uuid.uuid4())
            sessions[session_id] = {
                "created": time.strftime("%H:%M:%S"),
                "requests": 0,
                "label": f"session-{len(sessions)+1}",
            }
            print(f"\n  {YELLOW}New runtime session (will trigger cold start){RESET}")
            print(f"  {BLUE}Memory session unchanged:{RESET} {memory_session_id}")
            print(f"  {DIM}RAM history will be empty — durable memory will kick in{RESET}")
            print_status_bar(session_id, memory_session_id, user_id, users[user_id], 0)
            continue

        elif user_input == "/session":
            print(f"\n  {CYAN}Runtime session:{RESET} {session_id}")
            print(f"  {CYAN}Requests on this session:{RESET} {sessions[session_id]['requests']}")
            continue

        elif user_input == "/sessions":
            print(f"\n  {CYAN}All runtime sessions this run:{RESET}")
            for sid, info in sessions.items():
                marker = " <<< current" if sid == session_id else ""
                print(f"  {DIM}*{RESET} {sid}  ({info['created']}, {info['requests']} reqs){GREEN}{marker}{RESET}")
            continue

        elif user_input.startswith("/switch "):
            target = user_input[8:].strip()
            if target in sessions:
                session_id = target
                print(f"\n  {YELLOW}Switched to runtime session:{RESET} {session_id}")
                print_status_bar(session_id, memory_session_id, user_id, users[user_id], sessions[session_id]['requests'])
            else:
                matches = [s for s in sessions if s.startswith(target)]
                if len(matches) == 1:
                    session_id = matches[0]
                    print(f"\n  {YELLOW}Switched to runtime session:{RESET} {session_id}")
                    print_status_bar(session_id, memory_session_id, user_id, users[user_id], sessions[session_id]['requests'])
                else:
                    print(f"  {RED}Session not found. Use /sessions to list.{RESET}")
            continue

        elif user_input.startswith("/user "):
            new_uid = user_input[6:].strip()
            if new_uid in users:
                user_id = new_uid
                print(f"\n  {YELLOW}Switched to user:{RESET} {users[user_id]} (id={user_id})")
                print(f"  {DIM}Memory is isolated per user — different user = different history{RESET}")
            else:
                print(f"  {RED}Invalid user. Choose 1-5: {', '.join(f'{k}={v}' for k,v in users.items())}{RESET}")
            continue

        elif user_input == "/msession":
            print(f"\n  {BLUE}Memory ID:{RESET}        {memory_id}")
            print(f"  {BLUE}Memory session:{RESET}   {memory_session_id}")
            print(f"  {BLUE}Actor ID:{RESET}         {user_id} ({users[user_id]})")
            print(f"  {CYAN}Runtime session (client):{RESET} {session_id}")
            print(f"  {CYAN}Runtime session (agent):{RESET}  {last_agent_session_id}")
            print(f"  {DIM}Memory session is independent from runtime session — survives /new{RESET}")
            continue

        elif user_input.startswith("/msession "):
            new_msid = user_input[10:].strip()
            if new_msid:
                memory_session_id = new_msid
                print(f"\n  {BLUE}Memory session changed to:{RESET} {memory_session_id}")
                print(f"  {DIM}Conversation context will come from this memory session now{RESET}")
            else:
                print(f"  {RED}Usage: /msession <id>{RESET}")
            continue

        elif user_input == "/memory":
            if not memory_id:
                print(f"  {RED}No memory_id configured.{RESET}")
                continue
            print(f"\n  {DIM}Querying durable memory (user={user_id}, msession={memory_session_id})...{RESET}")
            try:
                latency_ms, response = invoke(
                    runtime,
                    {
                        "mode": "memory_query",
                        "memory_id": memory_id,
                        "user_id": user_id,
                        "memory_session_id": memory_session_id,
                        "k": 5,
                    },
                    session_id,
                )
                # Parse response — try multiple extraction strategies
                data = None
                response_str = str(response)

                # Strategy 1: dict with "response" key
                if isinstance(response, dict) and "response" in response:
                    parts = response["response"]
                    text = parts[0] if isinstance(parts, list) else str(parts)
                    try:
                        data = json.loads(str(text))
                    except (json.JSONDecodeError, TypeError):
                        pass

                # Strategy 2: find JSON in string representation
                if data is None and '{"turns"' in response_str:
                    try:
                        start = response_str.index('{"turns"')
                        depth = 0
                        for i in range(start, len(response_str)):
                            if response_str[i] == '{':
                                depth += 1
                            elif response_str[i] == '}':
                                depth -= 1
                                if depth == 0:
                                    data = json.loads(response_str[start:i + 1])
                                    break
                    except (ValueError, json.JSONDecodeError):
                        pass

                # Strategy 3: try parsing whole string
                if data is None:
                    try:
                        data = json.loads(response_str)
                    except (json.JSONDecodeError, TypeError):
                        pass

                if data is None:
                    data = {"error": f"Could not parse response: {response_str[:200]}", "turns": []}

                if data.get("error"):
                    print(f"  {RED}Error:{RESET} {data['error']}")
                elif not data.get("turns"):
                    print(f"  {DIM}No turns stored in durable memory for this user/session.{RESET}")
                else:
                    print(f"\n  {BLUE}Durable Memory — Last {data.get('count', '?')} turns:{RESET}")
                    print(f"  {DIM}{'─' * 55}{RESET}")
                    for i, turn in enumerate(data["turns"], 1):
                        print(f"  {CYAN}Turn {i}:{RESET}")
                        for msg in turn:
                            role = msg.get("role", "?")
                            text = msg.get("content", {}).get("text", "")
                            if len(text) > 120:
                                text = text[:120] + "..."
                            role_color = GREEN if role == "USER" else MAGENTA
                            print(f"    {role_color}{role}:{RESET} {text}")
                    print(f"  {DIM}{'─' * 55}{RESET}")
                print(f"  {DIM}Latency: {latency_ms:.0f} ms{RESET}")
            except Exception as e:
                print(f"  {RED}Error:{RESET} {e}")
            continue

        elif user_input == "/forget":
            old_msid = memory_session_id
            memory_session_id = str(uuid.uuid4())[:12]
            print(f"\n  {YELLOW}Memory cleared!{RESET}")
            print(f"  {DIM}Old memory session:{RESET} {old_msid}")
            print(f"  {BLUE}New memory session:{RESET} {memory_session_id}")
            print(f"  {DIM}Agent will not see previous conversation history.{RESET}")
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
                    mem_enabled = ping.get("memory_enabled", False)
                    print(f"\n  {BOLD}PONG{RESET}")
                    print(f"  ├─ {CYAN}Latency:{RESET}      {latency_ms:.0f} ms")
                    print(f"  ├─ {CYAN}Cache:{RESET}        {cache_color}{cache}{RESET}")
                    print(f"  ├─ {CYAN}Server time:{RESET}  {ping.get('elapsed_ms', '?')} ms")
                    print(f"  ├─ {CYAN}Invocations:{RESET} {ping.get('invocation_count', '?')}")
                    print(f"  ├─ {CYAN}Init time:{RESET}    {ping.get('init_time', '?')}")
                    print(f"  ├─ {CYAN}Session:{RESET}      {ping.get('session_id', '?')}")
                    print(f"  └─ {BLUE}Memory:{RESET}       {'enabled' if mem_enabled else 'disabled'}")
                else:
                    print(f"  {YELLOW}Response:{RESET} {str(response)[:200]}")
                    print(f"  {CYAN}Latency:{RESET} {latency_ms:.0f} ms")

                latency_log.append({
                    "type": "ping",
                    "session": session_id[:12],
                    "latency_ms": round(latency_ms),
                    "cache": ping.get("cache_status", "?") if ping else "?",
                    "memory_src": "-",
                })
            except Exception as e:
                print(f"  {RED}Error:{RESET} {e}")
            continue

        elif user_input == "/latency":
            if not latency_log:
                print(f"  {DIM}No requests yet.{RESET}")
                continue
            print(f"\n  {CYAN}{'#':<4} {'Type':<8} {'Session':<14} {'Latency':>10} {'Cache':<12} {'Memory':<12}{RESET}")
            print(f"  {DIM}{'─' * 64}{RESET}")
            for i, entry in enumerate(latency_log):
                cache_color = RED if entry.get('cache') == 'COLD_START' else GREEN if entry.get('cache') == 'WARM' else ""
                print(f"  {i+1:<4} {entry['type']:<8} {entry['session']:<14} {entry['latency_ms']:>8} ms {cache_color}{entry.get('cache', ''):<12}{RESET} {entry.get('memory_src', '')}")
            continue

        elif user_input == "/clear":
            os.system("clear")
            print_header()
            print_status_bar(session_id, memory_session_id, user_id, users[user_id], sessions[session_id]['requests'])
            continue

        elif user_input.startswith("/"):
            print(f"  {RED}Unknown command. Type /help{RESET}")
            continue

        # ── Chat mode (LLM call with hybrid memory) ──
        print(f"\n  {DIM}Sending to session {session_id}...{RESET}")
        try:
            payload = {
                "prompt": user_input,
                "user_id": user_id,
                "memory_id": memory_id,
                "memory_session_id": memory_session_id,
                "stream": False,
            }
            latency_ms, response = invoke(runtime, payload, session_id)
            request_count += 1
            sessions[session_id]['requests'] += 1

            response_text, memory_source, cache_status, agent_session_id = extract_response_text(response)
            # Track latest agent session ID for /msession display
            last_agent_session_id = agent_session_id

            print(f"\n  {MAGENTA}Agent:{RESET} {response_text}")
            print(f"\n  {DIM}Runtime session (client): {session_id}{RESET}")
            print(f"  {DIM}Runtime session (agent):  {agent_session_id}{RESET}")
            print(f"  {DIM}Memory session:           {memory_session_id}{RESET}")
            print(f"  {DIM}Latency: {latency_ms:.0f} ms | "
                  f"Memory: {memory_source_label(memory_source)} | "
                  f"Cache: {cache_status} | "
                  f"Requests: {sessions[session_id]['requests']}{RESET}")

            latency_log.append({
                "type": "chat",
                "session": session_id[:12],
                "latency_ms": round(latency_ms),
                "cache": cache_status,
                "memory_src": memory_source,
            })

        except Exception as e:
            print(f"  {RED}Error:{RESET} {e}")


if __name__ == "__main__":
    main()
