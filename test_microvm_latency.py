"""
MicroVM Latency & Session Concept Tests (v2)
==============================================
Properly isolates microVM overhead from LLM latency.

Tests:
  1-4:  PING mode (no LLM) — pure microVM cold/warm measurement
  5-7:  CHAT mode — LLM tests with conversation history & assertions
  8-9:  Concurrency tests
  10:   Latency breakdown summary

Prerequisites:
    - Run deploy.py first to deploy the agent
    - AWS credentials configured
    - deploy_info.json must exist
"""

import uuid
import json
import time
import threading
import concurrent.futures
from dataclasses import dataclass, field
from bedrock_agentcore_starter_toolkit import Runtime
from boto3.session import Session

# Lock to serialize Runtime.configure() calls (writes shared .yaml file)
_runtime_lock = threading.Lock()


@dataclass
class TestResult:
    test_name: str
    session_id: str
    latency_ms: float
    response: str = ""
    success: bool = True
    error: str = ""
    category: str = ""  # "ping" or "chat"


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
            print(f"  [{status}] {r.test_name}")
            print(f"         Session: {r.session_id}")
            print(f"         Latency: {r.latency_ms:.0f} ms")
            if r.error:
                print(f"         Error: {r.error}")

        # Latency table by category
        for cat in ["ping", "chat", "concurrent"]:
            cat_results = [r for r in self.results if r.category == cat]
            if not cat_results:
                continue
            label = {"ping": "PING (pure microVM)", "chat": "CHAT (LLM + microVM)", "concurrent": "CONCURRENT"}[cat]
            print(f"\n{'─' * 60}")
            print(f"  {label}")
            print(f"{'─' * 60}")
            print(f"  {'Test':<43} {'Latency':>10}")
            print(f"  {'─' * 53}")
            for r in cat_results:
                ms = f"{r.latency_ms:.0f} ms"
                print(f"  {r.test_name:<43} {ms:>10}")

    def print_breakdown(self):
        """Print the key insight: microVM overhead vs LLM latency."""
        ping_cold = [r for r in self.results if r.category == "ping" and "Cold" in r.test_name and r.success]
        ping_warm = [r for r in self.results if r.category == "ping" and "Warm" in r.test_name and r.success]
        chat_cold = [r for r in self.results if r.category == "chat" and "Cold" in r.test_name and r.success]
        chat_warm = [r for r in self.results if r.category == "chat" and "Warm" in r.test_name and r.success]

        print("\n" + "=" * 60)
        print("LATENCY BREAKDOWN — What You're Actually Measuring")
        print("=" * 60)

        if ping_cold and ping_warm:
            avg_ping_cold = sum(r.latency_ms for r in ping_cold) / len(ping_cold)
            avg_ping_warm = sum(r.latency_ms for r in ping_warm) / len(ping_warm)
            microvm_overhead = avg_ping_cold - avg_ping_warm
            print(f"\n  Ping Cold (microVM + cache load):  {avg_ping_cold:>8.0f} ms")
            print(f"  Ping Warm (microVM reused):         {avg_ping_warm:>8.0f} ms")
            print(f"  ─────────────────────────────────────────────")
            print(f"  Pure microVM cold start overhead:   {microvm_overhead:>8.0f} ms")

        if chat_cold and chat_warm:
            avg_chat_cold = sum(r.latency_ms for r in chat_cold) / len(chat_cold)
            avg_chat_warm = sum(r.latency_ms for r in chat_warm) / len(chat_warm)
            print(f"\n  Chat Cold (microVM + LLM):          {avg_chat_cold:>8.0f} ms")
            print(f"  Chat Warm (LLM only):               {avg_chat_warm:>8.0f} ms")

            if ping_warm:
                avg_ping_warm = sum(r.latency_ms for r in ping_warm) / len(ping_warm)
                llm_latency = avg_chat_warm - avg_ping_warm
                print(f"\n  ─── Estimated Component Breakdown ───")
                print(f"  Network/framework overhead:         {avg_ping_warm:>8.0f} ms")
                print(f"  LLM inference:                      {llm_latency:>8.0f} ms")
                if ping_cold:
                    avg_ping_cold = sum(r.latency_ms for r in ping_cold) / len(ping_cold)
                    print(f"  microVM cold start penalty:          {avg_ping_cold - avg_ping_warm:>8.0f} ms")

        print("=" * 60)


def timed_invoke(runtime: Runtime, payload: dict, session_id: str) -> tuple[float, str]:
    """Invoke the agent and return (latency_ms, response_text)."""
    start = time.perf_counter()
    response = runtime.invoke(payload, session_id=session_id)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, str(response)


def get_runtime() -> Runtime:
    """Initialize the Runtime from saved deployment info.
    Thread-safe: serializes configure() calls to avoid .yaml write races."""
    with _runtime_lock:
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


def parse_ping_response(response_str: str) -> dict | None:
    """Extract the JSON payload from a ping response."""
    try:
        # Response is wrapped in SDK response dict, find the JSON body
        if '"pong"' in response_str:
            start = response_str.index('{"pong"')
            # Find matching closing brace
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


# ===========================================================================
# PING TESTS — Pure microVM overhead (no LLM)
# ===========================================================================

def test_ping_cold(runtime: Runtime, suite: TestSuite):
    """PING with new session. Measures: network + microVM provision + cache load (1.2s)."""
    print("\n[TEST 1] Ping Cold Start (no LLM)")
    print("-" * 40)

    session_id = str(uuid.uuid4())
    try:
        latency_ms, response = timed_invoke(
            runtime, {"mode": "ping"}, session_id
        )
        ping = parse_ping_response(response)
        cache = ping.get("cache_status", "?") if ping else "?"
        invocations = ping.get("invocation_count", "?") if ping else "?"
        print(f"  Session: {session_id}...")
        print(f"  Latency: {latency_ms:.0f} ms")
        print(f"  Cache: {cache}, Invocations: {invocations}")

        success = ping is not None and ping.get("cache_status") == "COLD_START"
        if not success:
            print(f"  ASSERTION FAILED: Expected COLD_START, got {cache}")
        suite.add(TestResult("Ping Cold Start", session_id, latency_ms, response, success, category="ping"))
        return session_id
    except Exception as e:
        suite.add(TestResult("Ping Cold Start", session_id, 0, success=False, error=str(e), category="ping"))
        return None


def test_ping_warm(runtime: Runtime, suite: TestSuite, session_id: str, n: int = 3):
    """PING same session N times. Should be WARM + near-zero server time."""
    print(f"\n[TEST 2] Ping Warm Start x{n} (same session, no LLM)")
    print("-" * 40)

    for i in range(n):
        try:
            latency_ms, response = timed_invoke(
                runtime, {"mode": "ping"}, session_id
            )
            ping = parse_ping_response(response)
            cache = ping.get("cache_status", "?") if ping else "?"
            invocations = ping.get("invocation_count", "?") if ping else "?"
            server_ms = ping.get("elapsed_ms", "?") if ping else "?"
            print(f"  Warm #{i + 1}: {latency_ms:.0f} ms (server: {server_ms} ms, cache: {cache}, invocations: {invocations})")

            success = ping is not None and ping.get("cache_status") == "WARM"
            if not success:
                print(f"  ASSERTION FAILED: Expected WARM, got {cache}")
            suite.add(TestResult(f"Ping Warm #{i + 1}", session_id, latency_ms, response, success, category="ping"))
        except Exception as e:
            suite.add(TestResult(f"Ping Warm #{i + 1}", session_id, 0, success=False, error=str(e), category="ping"))


def test_ping_new_session_is_cold(runtime: Runtime, suite: TestSuite):
    """PING with a brand new session. Should be COLD_START again — proves isolation."""
    print("\n[TEST 3] Ping New Session = Cold Again (proves microVM isolation)")
    print("-" * 40)

    session_id = str(uuid.uuid4())
    try:
        latency_ms, response = timed_invoke(
            runtime, {"mode": "ping"}, session_id
        )
        ping = parse_ping_response(response)
        cache = ping.get("cache_status", "?") if ping else "?"
        invocations = ping.get("invocation_count", "?") if ping else "?"
        print(f"  Session: {session_id}... (NEW)")
        print(f"  Latency: {latency_ms:.0f} ms")
        print(f"  Cache: {cache}, Invocations: {invocations}")

        success = ping is not None and ping.get("cache_status") == "COLD_START" and ping.get("invocation_count") == 1
        if not success:
            print(f"  ASSERTION FAILED: Expected COLD_START + invocation_count=1")
        suite.add(TestResult("Ping Cold (new session)", session_id, latency_ms, response, success, category="ping"))
    except Exception as e:
        suite.add(TestResult("Ping Cold (new session)", session_id, 0, success=False, error=str(e), category="ping"))


def test_ping_invocation_counter(runtime: Runtime, suite: TestSuite, session_id: str):
    """Verify invocation counter increments — proves same microVM is reused."""
    print("\n[TEST 4] Ping Invocation Counter (proves same microVM)")
    print("-" * 40)

    try:
        latency_ms, response = timed_invoke(
            runtime, {"mode": "ping"}, session_id
        )
        ping = parse_ping_response(response)
        count = ping.get("invocation_count", 0) if ping else 0
        print(f"  Invocation count: {count}")
        print(f"  Latency: {latency_ms:.0f} ms")

        # After cold + 3 warm + this = at least 5
        success = ping is not None and count >= 5
        if not success:
            print(f"  ASSERTION FAILED: Expected invocation_count >= 5, got {count}")
        suite.add(TestResult("Ping Counter Check", session_id, latency_ms, response, success, category="ping"))
    except Exception as e:
        suite.add(TestResult("Ping Counter Check", session_id, 0, success=False, error=str(e), category="ping"))


# ===========================================================================
# CHAT TESTS — LLM calls with proper assertions
# ===========================================================================

def test_chat_cold(runtime: Runtime, suite: TestSuite):
    """Chat cold start: new session, LLM call. Baseline = microVM + LLM."""
    print("\n[TEST 5] Chat Cold Start (new session + LLM)")
    print("-" * 40)

    session_id = str(uuid.uuid4())
    try:
        latency_ms, response = timed_invoke(
            runtime,
            {"prompt": "What is 10 * 5?", "user_id": "1"},
            session_id
        )
        print(f"  Session: {session_id}...")
        print(f"  Latency: {latency_ms:.0f} ms")
        print(f"  Response preview: {response[:200]}...")

        # Check response mentions 50
        has_50 = "50" in response
        if not has_50:
            print(f"  WARNING: Response doesn't contain '50'")
        suite.add(TestResult("Chat Cold Start", session_id, latency_ms, response, has_50, category="chat"))
        return session_id
    except Exception as e:
        suite.add(TestResult("Chat Cold Start", session_id, 0, success=False, error=str(e), category="chat"))
        return None


def test_chat_warm(runtime: Runtime, suite: TestSuite, session_id: str):
    """Chat warm: same session, second LLM call."""
    print("\n[TEST 6] Chat Warm Start (same session + LLM)")
    print("-" * 40)

    try:
        latency_ms, response = timed_invoke(
            runtime,
            {"prompt": "What is 7 + 7?", "user_id": "1"},
            session_id
        )
        print(f"  Session: {session_id}... (reused)")
        print(f"  Latency: {latency_ms:.0f} ms")
        print(f"  Response preview: {response[:200]}...")

        has_14 = "14" in response
        if not has_14:
            print(f"  WARNING: Response doesn't contain '14'")
        suite.add(TestResult("Chat Warm Start", session_id, latency_ms, response, has_14, category="chat"))
    except Exception as e:
        suite.add(TestResult("Chat Warm Start", session_id, 0, success=False, error=str(e), category="chat"))


def test_chat_continuity(runtime: Runtime, suite: TestSuite, session_id: str):
    """Follow-up 'and that plus 34?' — history in RAM should give 48 (14+34)."""
    print("\n[TEST 7] Chat Session Continuity (history in RAM)")
    print("-" * 40)

    try:
        latency_ms, response = timed_invoke(
            runtime,
            {"prompt": "and that plus 34?", "user_id": "1"},
            session_id
        )
        print(f"  Session: {session_id}... (reused)")
        print(f"  Latency: {latency_ms:.0f} ms")
        print(f"  Response preview: {response[:300]}...")

        # Should contain 48 (14 + 34) if history is working
        has_48 = "48" in response
        if has_48:
            print(f"  ASSERTION PASSED: Response contains '48' (14+34)")
        else:
            print(f"  ASSERTION FAILED: Response doesn't contain '48' — history may not be working")
        suite.add(TestResult("Chat Continuity (14+34=48)", session_id, latency_ms, response, has_48, category="chat"))
    except Exception as e:
        suite.add(TestResult("Chat Continuity (14+34=48)", session_id, 0, success=False, error=str(e), category="chat"))


def test_chat_isolation(runtime: Runtime, suite: TestSuite):
    """New session — ask about a previous result that doesn't exist here.
    Verifies the new microVM has no RAM history from the other session."""
    print("\n[TEST 8] Chat Session Isolation (new session, no history)")
    print("-" * 40)

    session_id = str(uuid.uuid4())
    try:
        latency_ms, response = timed_invoke(
            runtime,
            {"prompt": "What was the last number we calculated together?", "user_id": "1"},
            session_id
        )
        print(f"  Session: {session_id} (NEW)")
        print(f"  Latency: {latency_ms:.0f} ms")
        print(f"  Response preview: {response[:300]}...")

        # The model should NOT know "48" or "14" from the other session's history
        leaked = "48" in response and "14" in response
        if not leaked:
            print(f"  ASSERTION PASSED: No prior computation leaked from other session")
        else:
            print(f"  ASSERTION FAILED: Response contains both '48' and '14' — history leaked across sessions")
        suite.add(TestResult("Chat Isolation (no history)", session_id, latency_ms, response, not leaked, category="chat"))
    except Exception as e:
        suite.add(TestResult("Chat Isolation (no history)", session_id, 0, success=False, error=str(e), category="chat"))


# ===========================================================================
# CONCURRENCY TESTS
# ===========================================================================

def test_concurrent_ping_same_session(runtime: Runtime, suite: TestSuite, n: int = 3):
    """N concurrent pings on same session.
    Each thread gets its own Runtime to avoid shared-state race condition."""
    print(f"\n[TEST 9] Concurrent Ping x{n} (same session)")
    print("-" * 40)

    session_id = str(uuid.uuid4())

    def invoke_one(idx):
        thread_runtime = get_runtime()
        return idx, timed_invoke(thread_runtime, {"mode": "ping"}, session_id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
        futures = {executor.submit(invoke_one, i): i for i in range(n)}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                idx, (latency_ms, response) = future.result()
                print(f"  Concurrent ping #{idx}: {latency_ms:.0f} ms")
                suite.add(TestResult(f"Concurrent Ping #{idx} (same)", session_id, latency_ms, response, category="concurrent"))
            except Exception as e:
                suite.add(TestResult(f"Concurrent Ping #{idx} (same)", session_id, 0, success=False, error=str(e), category="concurrent"))


def test_concurrent_ping_diff_sessions(runtime: Runtime, suite: TestSuite, n: int = 3):
    """N concurrent pings, each on a different session (parallel cold starts).
    Each thread gets its own Runtime to avoid 'no default agent' race condition."""
    print(f"\n[TEST 10] Concurrent Ping x{n} (different sessions)")
    print("-" * 40)

    def invoke_one(idx):
        # Each thread gets its own Runtime to avoid shared-state race condition
        thread_runtime = get_runtime()
        sid = str(uuid.uuid4())
        return idx, sid, timed_invoke(thread_runtime, {"mode": "ping"}, sid)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
        futures = {executor.submit(invoke_one, i): i for i in range(n)}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                idx, sid, (latency_ms, response) = future.result()
                print(f"  Concurrent ping #{idx} (session {sid}...): {latency_ms:.0f} ms")
                suite.add(TestResult(f"Concurrent Ping #{idx} (new)", sid, latency_ms, response, category="concurrent"))
            except Exception as e:
                suite.add(TestResult(f"Concurrent Ping #{idx} (new)", "unknown", 0, success=False, error=str(e), category="concurrent"))


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 80)
    print("AgentCore MicroVM Latency & Session Tests (v2)")
    print("Isolates microVM overhead from LLM latency")
    print("=" * 80)

    runtime = get_runtime()
    suite = TestSuite()

    # ── PING TESTS (pure microVM, no LLM) ──
    ping_session = test_ping_cold(runtime, suite)
    if ping_session:
        test_ping_warm(runtime, suite, ping_session, n=3)
        test_ping_new_session_is_cold(runtime, suite)
        test_ping_invocation_counter(runtime, suite, ping_session)

    # ── CHAT TESTS (LLM + assertions) ──
    chat_session = test_chat_cold(runtime, suite)
    if chat_session:
        test_chat_warm(runtime, suite, chat_session)
        test_chat_continuity(runtime, suite, chat_session)

    test_chat_isolation(runtime, suite)

    # ── CONCURRENCY TESTS ──
    test_concurrent_ping_same_session(runtime, suite, n=3)
    test_concurrent_ping_diff_sessions(runtime, suite, n=3)

    # ── RESULTS ──
    suite.print_summary()
    suite.print_breakdown()

    # Save
    results_json = [
        {
            "test": r.test_name,
            "category": r.category,
            "session_id": r.session_id,
            "latency_ms": r.latency_ms,
            "success": r.success,
            "error": r.error,
            "response_preview": r.response[:200] if r.response else "",
        }
        for r in suite.results
    ]
    with open("test_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print("\nResults saved to test_results.json")


if __name__ == "__main__":
    main()
