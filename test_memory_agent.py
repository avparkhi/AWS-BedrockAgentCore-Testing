"""
Memory Agent Tests
==================
Tests hybrid memory (RAM + durable), session independence, and user isolation.

Tests:
  1-4:  PING tests — microVM cold/warm (same as original)
  5:    Chat cold start — first message, memory_source = none
  6:    Chat warm — same session, memory_source = ram
  7:    Chat continuity — follow-up math, RAM history works
  8:    Memory persistence — new runtime session, durable memory loads previous context
  9:    User isolation — different user sees no history from other user
  10:   Memory query — /memory mode returns stored turns
  11:   Latency breakdown

Prerequisites:
    - Run deploy_memory.py first
    - deploy_memory_info.json must exist
"""

import uuid
import json
import time
import re
from dataclasses import dataclass, field
from bedrock_agentcore_starter_toolkit import Runtime


@dataclass
class TestResult:
    test_name: str
    session_id: str
    latency_ms: float
    response: str = ""
    success: bool = True
    error: str = ""
    category: str = ""
    memory_source: str = ""


@dataclass
class TestSuite:
    results: list = field(default_factory=list)

    def add(self, result: TestResult):
        self.results.append(result)

    def print_summary(self):
        print("\n" + "=" * 80)
        print("TEST RESULTS SUMMARY")
        print("=" * 80)

        passed = sum(1 for r in self.results if r.success)
        failed = sum(1 for r in self.results if not r.success)
        print(f"  {passed} passed, {failed} failed, {len(self.results)} total\n")

        for r in self.results:
            status = "PASS" if r.success else "FAIL"
            mem = f" [memory: {r.memory_source}]" if r.memory_source else ""
            print(f"  [{status}] {r.test_name}{mem}")
            print(f"         Latency: {r.latency_ms:.0f} ms")
            if r.error:
                print(f"         Error: {r.error}")

        # Latency table by category
        for cat in ["ping", "chat", "memory"]:
            cat_results = [r for r in self.results if r.category == cat]
            if not cat_results:
                continue
            label = {
                "ping": "PING (pure microVM)",
                "chat": "CHAT (LLM + memory)",
                "memory": "MEMORY (durable persistence)",
            }[cat]
            print(f"\n{'─' * 65}")
            print(f"  {label}")
            print(f"{'─' * 65}")
            print(f"  {'Test':<45} {'Latency':>10} {'Memory':>8}")
            print(f"  {'─' * 63}")
            for r in cat_results:
                ms = f"{r.latency_ms:.0f} ms"
                print(f"  {r.test_name:<45} {ms:>10} {r.memory_source:>8}")


def timed_invoke(runtime, payload, session_id):
    start = time.perf_counter()
    response = runtime.invoke(payload, session_id=session_id)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, response


def get_runtime():
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


def extract_text_and_meta(response):
    """Extract response text and META tag from agent response."""
    raw = ""
    if isinstance(response, dict) and "response" in response:
        parts = response["response"]
        if isinstance(parts, list):
            raw = "\n".join(str(p) for p in parts)
        else:
            raw = str(parts)
    else:
        raw = str(response)

    # Decode escaped characters
    if raw.startswith('"') and raw.endswith('"'):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
    if '\\n' in raw:
        raw = raw.replace('\\n', '\n').replace('\\t', '\t')
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1]

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
        text = re.sub(r"<<META:.*?>>\n?", "", raw).strip()
    else:
        text = raw.strip()

    return text, memory_source, cache_status, agent_session_id


# ===========================================================================
# PING TESTS
# ===========================================================================

def test_ping_cold(runtime, suite):
    print("\n[TEST 1] Ping Cold Start")
    print("-" * 40)
    session_id = str(uuid.uuid4())
    try:
        latency_ms, response = timed_invoke(runtime, {"mode": "ping", "stream": False}, session_id)
        ping = parse_ping_response(response)
        cache = ping.get("cache_status", "?") if ping else "?"
        mem_enabled = ping.get("memory_enabled", False) if ping else False
        print(f"  Latency: {latency_ms:.0f} ms | Cache: {cache} | Memory enabled: {mem_enabled}")

        success = ping is not None and cache == "COLD_START" and mem_enabled
        if not success:
            print(f"  ASSERTION FAILED: Expected COLD_START + memory_enabled=True")
        suite.add(TestResult("Ping Cold Start", session_id, latency_ms, str(response), success, category="ping"))
        return session_id
    except Exception as e:
        suite.add(TestResult("Ping Cold Start", session_id, 0, success=False, error=str(e), category="ping"))
        return None


def test_ping_warm(runtime, suite, session_id):
    print("\n[TEST 2] Ping Warm Start x3")
    print("-" * 40)
    for i in range(3):
        try:
            latency_ms, response = timed_invoke(runtime, {"mode": "ping", "stream": False}, session_id)
            ping = parse_ping_response(response)
            cache = ping.get("cache_status", "?") if ping else "?"
            print(f"  Warm #{i+1}: {latency_ms:.0f} ms | Cache: {cache}")
            success = ping is not None and cache == "WARM"
            suite.add(TestResult(f"Ping Warm #{i+1}", session_id, latency_ms, str(response), success, category="ping"))
        except Exception as e:
            suite.add(TestResult(f"Ping Warm #{i+1}", session_id, 0, success=False, error=str(e), category="ping"))


def test_ping_new_session_cold(runtime, suite):
    print("\n[TEST 3] Ping New Session = Cold Again")
    print("-" * 40)
    session_id = str(uuid.uuid4())
    try:
        latency_ms, response = timed_invoke(runtime, {"mode": "ping", "stream": False}, session_id)
        ping = parse_ping_response(response)
        cache = ping.get("cache_status", "?") if ping else "?"
        print(f"  Latency: {latency_ms:.0f} ms | Cache: {cache}")
        success = ping is not None and cache == "COLD_START"
        suite.add(TestResult("Ping Cold (new session)", session_id, latency_ms, str(response), success, category="ping"))
    except Exception as e:
        suite.add(TestResult("Ping Cold (new session)", session_id, 0, success=False, error=str(e), category="ping"))


# ===========================================================================
# CHAT TESTS — with memory
# ===========================================================================

def test_chat_cold_no_memory(runtime, suite, memory_id, memory_session_id):
    """First chat on a fresh memory session — memory_source should be 'none'."""
    print("\n[TEST 4] Chat Cold Start — No Prior Memory")
    print("-" * 40)
    session_id = str(uuid.uuid4())
    try:
        latency_ms, response = timed_invoke(
            runtime,
            {
                "prompt": "What is 10 * 5?",
                "user_id": "1",
                "memory_id": memory_id,
                "memory_session_id": memory_session_id,
                "stream": False,
            },
            session_id,
        )
        text, mem_src, cache, _ = extract_text_and_meta(response)
        print(f"  Latency: {latency_ms:.0f} ms | Memory: {mem_src} | Cache: {cache}")
        print(f"  Response: {text[:150]}...")

        has_50 = "50" in text
        is_none_memory = mem_src == "none"
        success = has_50 and is_none_memory
        if not has_50:
            print(f"  WARN: Response doesn't contain '50'")
        if not is_none_memory:
            print(f"  WARN: Expected memory_source=none, got {mem_src}")
        suite.add(TestResult("Chat Cold — No Memory", session_id, latency_ms, text, success, category="chat", memory_source=mem_src))
        return session_id
    except Exception as e:
        suite.add(TestResult("Chat Cold — No Memory", session_id, 0, success=False, error=str(e), category="chat"))
        return None


def test_chat_warm_ram(runtime, suite, session_id, memory_id, memory_session_id):
    """Second chat on same session — memory_source should be 'ram' (or 'both')."""
    print("\n[TEST 5] Chat Warm — RAM Memory")
    print("-" * 40)
    try:
        latency_ms, response = timed_invoke(
            runtime,
            {
                "prompt": "What is 7 + 7?",
                "user_id": "1",
                "memory_id": memory_id,
                "memory_session_id": memory_session_id,
                "stream": False,
            },
            session_id,
        )
        text, mem_src, cache, _ = extract_text_and_meta(response)
        print(f"  Latency: {latency_ms:.0f} ms | Memory: {mem_src} | Cache: {cache}")
        print(f"  Response: {text[:150]}...")

        has_14 = "14" in text
        has_ram = mem_src in ("ram", "both")
        success = has_14 and has_ram
        if not has_14:
            print(f"  WARN: Response doesn't contain '14'")
        if not has_ram:
            print(f"  WARN: Expected memory_source=ram or both, got {mem_src}")
        suite.add(TestResult("Chat Warm — RAM", session_id, latency_ms, text, success, category="chat", memory_source=mem_src))
    except Exception as e:
        suite.add(TestResult("Chat Warm — RAM", session_id, 0, success=False, error=str(e), category="chat"))


def test_chat_continuity(runtime, suite, session_id, memory_id, memory_session_id):
    """Follow-up math: 'and that plus 34?' — should get 48 (14+34) from RAM history."""
    print("\n[TEST 6] Chat Continuity — RAM Follow-up (14+34=48)")
    print("-" * 40)
    try:
        latency_ms, response = timed_invoke(
            runtime,
            {
                "prompt": "and that plus 34?",
                "user_id": "1",
                "memory_id": memory_id,
                "memory_session_id": memory_session_id,
                "stream": False,
            },
            session_id,
        )
        text, mem_src, cache, _ = extract_text_and_meta(response)
        print(f"  Latency: {latency_ms:.0f} ms | Memory: {mem_src} | Cache: {cache}")
        print(f"  Response: {text[:150]}...")

        has_48 = "48" in text
        success = has_48
        if has_48:
            print(f"  PASS: Contains '48' (14+34)")
        else:
            print(f"  FAIL: Response doesn't contain '48'")
        suite.add(TestResult("Chat Continuity (14+34=48)", session_id, latency_ms, text, success, category="chat", memory_source=mem_src))
    except Exception as e:
        suite.add(TestResult("Chat Continuity (14+34=48)", session_id, 0, success=False, error=str(e), category="chat"))


def test_memory_persistence(runtime, suite, memory_id, memory_session_id):
    """NEW runtime session — RAM is empty, durable memory should have previous context.
    This is the KEY test: proves memory survives across runtime sessions."""
    print("\n[TEST 7] Memory Persistence — New Session, Durable Memory Loads")
    print("-" * 40)
    session_id = str(uuid.uuid4())
    print(f"  New runtime session: {session_id[:12]}...")
    try:
        latency_ms, response = timed_invoke(
            runtime,
            {
                "prompt": "What was the result of 10 * 5 that we calculated earlier?",
                "user_id": "1",
                "memory_id": memory_id,
                "memory_session_id": memory_session_id,
                "stream": False,
            },
            session_id,
        )
        text, mem_src, cache, _ = extract_text_and_meta(response)
        print(f"  Latency: {latency_ms:.0f} ms | Memory: {mem_src} | Cache: {cache}")
        print(f"  Response: {text[:200]}...")

        has_50 = "50" in text
        from_durable = mem_src == "durable"
        success = has_50 and from_durable
        if has_50:
            print(f"  PASS: Remembers '50' from previous session")
        else:
            print(f"  FAIL: Doesn't mention '50'")
        if from_durable:
            print(f"  PASS: memory_source=durable (loaded from AgentCore Memory)")
        else:
            print(f"  WARN: Expected memory_source=durable, got {mem_src}")
        suite.add(TestResult("Memory Persistence (cross-session)", session_id, latency_ms, text, success, category="memory", memory_source=mem_src))
    except Exception as e:
        suite.add(TestResult("Memory Persistence (cross-session)", session_id, 0, success=False, error=str(e), category="memory"))


def test_user_isolation(runtime, suite, memory_id, memory_session_id):
    """Different user (user_id=2) should NOT see user_id=1's conversation history."""
    print("\n[TEST 8] User Isolation — User 2 Sees No History From User 1")
    print("-" * 40)
    session_id = str(uuid.uuid4())
    try:
        latency_ms, response = timed_invoke(
            runtime,
            {
                "prompt": "What calculations have we done together?",
                "user_id": "2",
                "memory_id": memory_id,
                "memory_session_id": memory_session_id,
                "stream": False,
            },
            session_id,
        )
        text, mem_src, cache, _ = extract_text_and_meta(response)
        print(f"  Latency: {latency_ms:.0f} ms | Memory: {mem_src} | Cache: {cache}")
        print(f"  Response: {text[:200]}...")

        # User 2 should not know about "50" or "48" from User 1's history
        leaked = "50" in text and "48" in text
        success = not leaked
        if success:
            print(f"  PASS: User 2 doesn't see User 1's calculations")
        else:
            print(f"  FAIL: User 1's history leaked to User 2")
        suite.add(TestResult("User Isolation (user 2)", session_id, latency_ms, text, success, category="memory", memory_source=mem_src))
    except Exception as e:
        suite.add(TestResult("User Isolation (user 2)", session_id, 0, success=False, error=str(e), category="memory"))


def test_memory_query(runtime, suite, session_id, memory_id, memory_session_id):
    """Test memory_query mode — should return stored turns without LLM call."""
    print("\n[TEST 9] Memory Query Mode — Returns Stored Turns")
    print("-" * 40)
    try:
        latency_ms, response = timed_invoke(
            runtime,
            {
                "mode": "memory_query",
                "memory_id": memory_id,
                "user_id": "1",
                "memory_session_id": memory_session_id,
                "k": 3,
                "stream": False,
            },
            session_id,
        )
        print(f"  Latency: {latency_ms:.0f} ms")

        # Parse response — same multi-strategy approach as chat_memory.py /memory
        data = None
        response_str = str(response)

        # Strategy 1: dict with "response" key — iteratively decode inner text
        if isinstance(response, dict) and "response" in response:
            parts = response["response"]
            text = parts[0] if isinstance(parts, list) else str(parts)
            decoded = str(text)
            for _ in range(4):
                if isinstance(decoded, dict):
                    data = decoded
                    break
                if not isinstance(decoded, str):
                    break
                try:
                    decoded = json.loads(decoded)
                except (json.JSONDecodeError, TypeError):
                    break
            if isinstance(decoded, dict):
                data = decoded

        # Strategy 2: find {"turns" in string and brace-match
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

        # Strategy 3: parse whole string
        if data is None:
            try:
                data = json.loads(response_str)
            except (json.JSONDecodeError, TypeError):
                pass

        if not data:
            print(f"  DEBUG: Could not parse response: {response_str[:200]}")

        if data:
            count = data.get("count", 0)
            has_turns = count > 0
            print(f"  Turns found: {count}")
            if has_turns:
                print(f"  PASS: Memory query returned {count} turns")
            else:
                print(f"  WARN: No turns returned (may be empty)")
            suite.add(TestResult("Memory Query Mode", session_id, latency_ms, str(data), has_turns, category="memory", memory_source="query"))
        else:
            print(f"  FAIL: Could not parse memory query response")
            suite.add(TestResult("Memory Query Mode", session_id, latency_ms, response_str[:200], False, error="Parse failed", category="memory"))
    except Exception as e:
        suite.add(TestResult("Memory Query Mode", session_id, 0, success=False, error=str(e), category="memory"))


def test_msession_isolation(runtime, suite, memory_id):
    """Different memory session should have no history from 'default' session."""
    print("\n[TEST 10] Memory Session Isolation — Different msession = No History")
    print("-" * 40)
    session_id = str(uuid.uuid4())
    isolated_msession = f"test-isolated-{uuid.uuid4().hex[:8]}"
    try:
        latency_ms, response = timed_invoke(
            runtime,
            {
                "prompt": "What calculations have we done together?",
                "user_id": "1",
                "memory_id": memory_id,
                "memory_session_id": isolated_msession,
                "stream": False,
            },
            session_id,
        )
        text, mem_src, cache, _ = extract_text_and_meta(response)
        print(f"  Memory session: {isolated_msession}")
        print(f"  Latency: {latency_ms:.0f} ms | Memory: {mem_src} | Cache: {cache}")
        print(f"  Response: {text[:200]}...")

        is_none = mem_src == "none"
        no_leak = "50" not in text and "48" not in text
        success = is_none and no_leak
        if is_none:
            print(f"  PASS: memory_source=none (fresh memory session)")
        else:
            print(f"  WARN: Expected memory_source=none, got {mem_src}")
        if no_leak:
            print(f"  PASS: No history leaked from 'default' session")
        else:
            print(f"  FAIL: History leaked across memory sessions")
        suite.add(TestResult("Memory Session Isolation", session_id, latency_ms, text, success, category="memory", memory_source=mem_src))
    except Exception as e:
        suite.add(TestResult("Memory Session Isolation", session_id, 0, success=False, error=str(e), category="memory"))


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 80)
    print("Memory Agent Tests — Hybrid RAM + Durable Memory")
    print("=" * 80)

    runtime, deploy_info = get_runtime()
    memory_id = deploy_info.get("memory_id")
    if not memory_id:
        print("ERROR: No memory_id in deploy_memory_info.json")
        return

    print(f"  Memory ID: {memory_id}")

    suite = TestSuite()

    # Use a unique memory session for this test run to avoid pollution
    test_memory_session = f"test-{uuid.uuid4().hex[:8]}"
    print(f"  Test memory session: {test_memory_session}")
    print()

    # ── PING TESTS ──
    ping_session = test_ping_cold(runtime, suite)
    if ping_session:
        test_ping_warm(runtime, suite, ping_session)
        test_ping_new_session_cold(runtime, suite)

    # ── CHAT TESTS (with memory) ──
    chat_session = test_chat_cold_no_memory(runtime, suite, memory_id, test_memory_session)
    if chat_session:
        test_chat_warm_ram(runtime, suite, chat_session, memory_id, test_memory_session)
        test_chat_continuity(runtime, suite, chat_session, memory_id, test_memory_session)

    # ── MEMORY PERSISTENCE TEST (the big one) ──
    test_memory_persistence(runtime, suite, memory_id, test_memory_session)

    # ── ISOLATION TESTS ──
    test_user_isolation(runtime, suite, memory_id, test_memory_session)
    test_msession_isolation(runtime, suite, memory_id)

    # ── MEMORY QUERY TEST ──
    # Use the ping_session (should still be warm)
    if ping_session:
        test_memory_query(runtime, suite, ping_session, memory_id, test_memory_session)

    # ── RESULTS ──
    suite.print_summary()

    # Save results
    results_json = [
        {
            "test": r.test_name,
            "category": r.category,
            "latency_ms": r.latency_ms,
            "success": r.success,
            "memory_source": r.memory_source,
            "error": r.error,
        }
        for r in suite.results
    ]
    with open("test_memory_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print("\nResults saved to test_memory_results.json")


if __name__ == "__main__":
    main()
