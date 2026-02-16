"""
Streaming Chat Interface for Memory Agent
==========================================
Uses boto3 directly to receive SSE streaming responses from the agent.
Tokens appear as they're generated — no waiting for full response.

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

Just type normally to chat with the agent (streaming mode).

Memory Architecture:
  - RAM history: Fast, same microVM session only (resets on /new)
  - Durable memory: Persists across sessions, restarts, redeploys
  - On warm starts: RAM used for context (fast, no API call)
  - On cold starts: Durable memory loaded automatically

Streaming:
  - Tokens appear as they're generated (~1-1.5s to first token)
  - Total time similar to non-streaming (~5-6s)
  - Uses boto3 directly (SDK runtime.invoke returns {} for streaming)
"""

import uuid
import json
import time
import os
import sys
import boto3
from botocore.config import Config

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


def get_agent_config():
    """Load agent config from deploy_memory_info.json and .bedrock_agentcore.yaml."""
    with open("deploy_memory_info.json") as f:
        deploy_info = json.load(f)

    region = deploy_info["region"]

    # Get agent ARN from the YAML config
    import yaml
    with open(".bedrock_agentcore.yaml") as f:
        config = yaml.safe_load(f)

    agent_config = config["agents"]["memory_agent"]
    agent_arn = agent_config["bedrock_agentcore"]["agent_arn"]

    return {
        "agent_arn": agent_arn,
        "region": region,
        "memory_id": deploy_info.get("memory_id"),
    }


def get_dataplane_client(region):
    """Create boto3 bedrock-agentcore dataplane client."""
    endpoint_url = os.getenv(
        "BEDROCK_AGENTCORE_DP_ENDPOINT",
        f"https://bedrock-agentcore.{region}.amazonaws.com",
    )
    config = Config(
        read_timeout=900,
        connect_timeout=60,
        retries={"max_attempts": 3},
    )
    return boto3.client(
        "bedrock-agentcore",
        region_name=region,
        endpoint_url=endpoint_url,
        config=config,
    )


def invoke_non_streaming(client, agent_arn, session_id, payload):
    """Invoke agent with stream=False, return parsed response."""
    payload["stream"] = False
    response = client.invoke_agent_runtime(
        agentRuntimeArn=agent_arn,
        runtimeSessionId=session_id,
        qualifier="DEFAULT",
        payload=json.dumps(payload, ensure_ascii=False),
    )

    content_type = response.get("contentType", "")

    # Non-streaming: read full response
    body = response.get("response")
    if body is None:
        return {}

    # EventStream response
    events = []
    try:
        for event in body:
            if isinstance(event, bytes):
                try:
                    decoded = event.decode("utf-8")
                    if decoded.startswith('"') and decoded.endswith('"'):
                        event = json.loads(decoded)
                    else:
                        event = decoded
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass
            events.append(event)
    except Exception as e:
        events = [f"Error reading EventStream: {e}"]

    return {"response": events}


def _parse_sse_chunk(json_text):
    """Parse a single SSE JSON chunk and yield (chunk_type, data) tuple."""
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        return

    if isinstance(parsed, dict):
        chunk_type = parsed.get("type")
        if chunk_type == "meta":
            yield ("meta", parsed)
        elif chunk_type == "done":
            yield ("done", parsed)
        elif chunk_type == "error":
            yield ("error", parsed.get("message", "Unknown error"))
        else:
            # Unknown dict — yield as text
            yield ("text", json.dumps(parsed, ensure_ascii=False))
    elif isinstance(parsed, str):
        yield ("text", parsed)


def invoke_streaming(client, agent_arn, session_id, payload):
    """Invoke agent with stream=True, yield SSE chunks.

    Yields tuples of (chunk_type, data):
      ("meta", dict)  - first chunk with metadata
      ("text", str)   - text token
      ("done", dict)  - final chunk with metadata
      ("error", str)  - error message
    """
    payload["stream"] = True
    response = client.invoke_agent_runtime(
        agentRuntimeArn=agent_arn,
        runtimeSessionId=session_id,
        qualifier="DEFAULT",
        payload=json.dumps(payload, ensure_ascii=False),
    )

    content_type = response.get("contentType", "")
    body = response.get("response")
    if body is None:
        return

    if "text/event-stream" not in content_type:
        # Fallback: not streaming, read as normal
        try:
            events = []
            for event in body:
                if isinstance(event, bytes):
                    try:
                        events.append(event.decode("utf-8"))
                    except UnicodeDecodeError:
                        pass
                else:
                    events.append(str(event))
            yield ("text", "".join(events))
        except Exception as e:
            yield ("error", str(e))
        return

    # SSE streaming — parse "data: <json>" lines
    # iter_lines is available on StreamingBody (boto3 blob with streaming=true)
    try:
        line_iter = body.iter_lines(chunk_size=1)
    except AttributeError:
        # Fallback: body is an EventStream, iterate events directly
        line_iter = None

    if line_iter is not None:
        for line in line_iter:
            if not line:
                continue
            decoded = line.decode("utf-8") if isinstance(line, bytes) else line
            if not decoded.startswith("data: "):
                continue
            yield from _parse_sse_chunk(decoded[6:])
    else:
        # EventStream fallback: events come as bytes without SSE framing
        for event in body:
            if isinstance(event, bytes):
                try:
                    decoded = event.decode("utf-8")
                except UnicodeDecodeError:
                    continue
            else:
                decoded = str(event)
            # May contain SSE framing or raw JSON
            for line in decoded.strip().split("\n"):
                line = line.strip()
                if line.startswith("data: "):
                    yield from _parse_sse_chunk(line[6:])
                elif line:
                    yield from _parse_sse_chunk(line)


def parse_ping_response(response) -> dict | None:
    """Extract ping JSON from response."""
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


def print_header():
    print(f"\n{BOLD}{'=' * 65}{RESET}")
    print(f"{BOLD}  AgentCore MicroVM — Streaming Memory Agent Chat{RESET}")
    print(f"{BOLD}{'=' * 65}{RESET}")
    print(f"  {DIM}Type /help for commands, or just type to chat{RESET}")
    print(f"  {DIM}Streaming: tokens appear as they're generated{RESET}")
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
    print(f"  {DIM}Loading agent configuration...{RESET}")

    try:
        agent_config = get_agent_config()
    except Exception as e:
        print(f"  {RED}Error loading config: {e}{RESET}")
        print(f"  {DIM}Make sure deploy_memory_info.json and .bedrock_agentcore.yaml exist{RESET}")
        return

    agent_arn = agent_config["agent_arn"]
    region = agent_config["region"]
    memory_id = agent_config["memory_id"]

    print(f"  {DIM}Creating boto3 client...{RESET}")
    client = get_dataplane_client(region)

    print(f"  {GREEN}Connected!{RESET}")
    print(f"  {CYAN}Agent ARN:{RESET} {agent_arn}")
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
                start = time.perf_counter()
                response = invoke_non_streaming(
                    client, agent_arn, session_id,
                    {
                        "mode": "memory_query",
                        "memory_id": memory_id,
                        "user_id": user_id,
                        "memory_session_id": memory_session_id,
                        "k": 5,
                    },
                )
                latency_ms = (time.perf_counter() - start) * 1000

                # Parse response
                data = None
                response_str = str(response)

                if isinstance(response, dict) and "response" in response:
                    parts = response["response"]
                    text = parts[0] if isinstance(parts, list) else str(parts)
                    try:
                        data = json.loads(str(text))
                    except (json.JSONDecodeError, TypeError):
                        pass

                if data is None and '{"turns"' in response_str:
                    try:
                        s = response_str.index('{"turns"')
                        depth = 0
                        for i in range(s, len(response_str)):
                            if response_str[i] == '{':
                                depth += 1
                            elif response_str[i] == '}':
                                depth -= 1
                                if depth == 0:
                                    data = json.loads(response_str[s:i + 1])
                                    break
                    except (ValueError, json.JSONDecodeError):
                        pass

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
                start = time.perf_counter()
                response = invoke_non_streaming(
                    client, agent_arn, session_id, {"mode": "ping"},
                )
                latency_ms = (time.perf_counter() - start) * 1000
                request_count += 1
                sessions[session_id]['requests'] += 1

                ping = parse_ping_response(response)
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
            print(f"\n  {CYAN}{'#':<4} {'Type':<8} {'Session':<14} {'Latency':>10} {'TTFT':>8} {'Cache':<12} {'Memory':<12}{RESET}")
            print(f"  {DIM}{'─' * 72}{RESET}")
            for i, entry in enumerate(latency_log):
                cache_color = RED if entry.get('cache') == 'COLD_START' else GREEN if entry.get('cache') == 'WARM' else ""
                ttft = f"{entry['ttft_ms']} ms" if entry.get('ttft_ms') else "-"
                print(f"  {i+1:<4} {entry['type']:<8} {entry['session']:<14} {entry['latency_ms']:>8} ms {ttft:>8} {cache_color}{entry.get('cache', ''):<12}{RESET} {entry.get('memory_src', '')}")
            continue

        elif user_input == "/clear":
            os.system("clear")
            print_header()
            print_status_bar(session_id, memory_session_id, user_id, users[user_id], sessions[session_id]['requests'])
            continue

        elif user_input.startswith("/"):
            print(f"  {RED}Unknown command. Type /help{RESET}")
            continue

        # ── Chat mode (streaming) ──
        print(f"\n  {DIM}Streaming from session {session_id}...{RESET}")
        try:
            payload = {
                "prompt": user_input,
                "user_id": user_id,
                "memory_id": memory_id,
                "memory_session_id": memory_session_id,
            }

            start = time.perf_counter()
            first_token_time = None
            memory_source = "?"
            cache_status = "?"
            agent_session_id = "?"
            ram_entries = "?"

            print(f"\n  {MAGENTA}Agent:{RESET} ", end="", flush=True)

            for chunk_type, data in invoke_streaming(client, agent_arn, session_id, payload):
                if chunk_type == "meta":
                    memory_source = data.get("memory_source", "?")
                    cache_status = data.get("cache_status", "?")
                    agent_session_id = data.get("session_id", "?")

                elif chunk_type == "text":
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    print(data, end="", flush=True)

                elif chunk_type == "done":
                    memory_source = data.get("memory_source", memory_source)
                    ram_entries = data.get("ram_entries", "?")

                elif chunk_type == "error":
                    print(f"\n  {RED}Stream error: {data}{RESET}")

            total_ms = (time.perf_counter() - start) * 1000
            ttft_ms = ((first_token_time - start) * 1000) if first_token_time else None
            request_count += 1
            sessions[session_id]['requests'] += 1
            last_agent_session_id = agent_session_id

            print()  # newline after streamed text
            print(f"\n  {DIM}Runtime session (client): {session_id}{RESET}")
            print(f"  {DIM}Runtime session (agent):  {agent_session_id}{RESET}")
            print(f"  {DIM}Memory session:           {memory_session_id}{RESET}")
            ttft_str = f"{ttft_ms:.0f} ms" if ttft_ms else "-"
            print(f"  {DIM}Total: {total_ms:.0f} ms | "
                  f"TTFT: {ttft_str} | "
                  f"Memory: {memory_source_label(memory_source)} | "
                  f"Cache: {cache_status} | "
                  f"RAM entries: {ram_entries}{RESET}")

            latency_log.append({
                "type": "chat",
                "session": session_id[:12],
                "latency_ms": round(total_ms),
                "ttft_ms": round(ttft_ms) if ttft_ms else None,
                "cache": cache_status,
                "memory_src": memory_source,
            })

        except Exception as e:
            print(f"\n  {RED}Error:{RESET} {e}")


if __name__ == "__main__":
    main()
