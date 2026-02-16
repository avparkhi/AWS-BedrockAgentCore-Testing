"""
Quick Latency Comparison Test
==============================
Focused test comparing cold start vs warm start latency.
Runs a minimal set of invocations and prints a clear comparison.

Usage:
    python test_quick_latency.py
"""

import uuid
import json
import time
from bedrock_agentcore_starter_toolkit import Runtime
from boto3.session import Session


def timed_invoke(runtime, payload, session_id):
    start = time.perf_counter()
    response = runtime.invoke(payload, session_id=session_id)
    elapsed = (time.perf_counter() - start) * 1000
    return elapsed, str(response)


def main():
    with open("deploy_info.json") as f:
        info = json.load(f)

    runtime = Runtime()
    boto_session = Session()
    runtime.configure(
        entrypoint="strands_claude_context.py",
        auto_create_execution_role=True,
        auto_create_ecr=True,
        requirements_file="requirements.txt",
        region=info["region"],
        agent_name="strands_claude_context"
    )

    print("=" * 60)
    print("Quick Latency Comparison: Cold vs Warm")
    print("=" * 60)

    # --- Cold start ---
    session_1 = str(uuid.uuid4())
    print(f"\n1. Cold Start (new session: {session_1[:12]}...)")
    cold_ms, cold_resp = timed_invoke(
        runtime, {"prompt": "What is 5 + 5?", "user_id": "1"}, session_1
    )
    print(f"   Latency: {cold_ms:.0f} ms")

    # --- Warm start (same session) ---
    print(f"\n2. Warm Start (same session: {session_1[:12]}...)")
    warm_ms, warm_resp = timed_invoke(
        runtime, {"prompt": "What is 10 + 10?", "user_id": "1"}, session_1
    )
    print(f"   Latency: {warm_ms:.0f} ms")

    # --- Another warm ---
    print(f"\n3. Warm Start #2 (same session: {session_1[:12]}...)")
    warm2_ms, warm2_resp = timed_invoke(
        runtime, {"prompt": "What is 20 + 20?", "user_id": "1"}, session_1
    )
    print(f"   Latency: {warm2_ms:.0f} ms")

    # --- New cold start ---
    session_2 = str(uuid.uuid4())
    print(f"\n4. Cold Start #2 (new session: {session_2[:12]}...)")
    cold2_ms, cold2_resp = timed_invoke(
        runtime, {"prompt": "What is 3 + 3?", "user_id": "2"}, session_2
    )
    print(f"   Latency: {cold2_ms:.0f} ms")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  Cold Start #1:  {cold_ms:>8.0f} ms")
    print(f"  Warm Start #1:  {warm_ms:>8.0f} ms")
    print(f"  Warm Start #2:  {warm2_ms:>8.0f} ms")
    print(f"  Cold Start #2:  {cold2_ms:>8.0f} ms")
    print(f"  ---")
    avg_warm = (warm_ms + warm2_ms) / 2
    avg_cold = (cold_ms + cold2_ms) / 2
    print(f"  Avg Cold:       {avg_cold:>8.0f} ms")
    print(f"  Avg Warm:       {avg_warm:>8.0f} ms")
    if avg_warm > 0:
        print(f"  Speedup:        {avg_cold / avg_warm:>8.1f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
