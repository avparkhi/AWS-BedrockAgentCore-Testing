"""
AgentCore Runtime Context Agent
Demonstrates session management, context handling, and microVM behavior.

Modes:
  - "ping": No LLM call. Returns instantly with cache status. Use to isolate microVM overhead.
  - "chat": Normal LLM call with conversation history tracked in RAM.
"""

import time
import json
from strands import Agent, tool
from strands_tools import calculator
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands.models import BedrockModel
from datetime import datetime

app = BedrockAgentCoreApp()

# ── Globals that persist within a microVM session ──
BIG_RULES = None
CONVERSATION_HISTORY = {}  # keyed by user_id, stores list of (role, message) tuples
INVOCATION_COUNT = 0
INIT_TIME = None


def load_heavy_rules():
    """Simulate loading expensive rules/config."""
    time.sleep(1.2)  # simulate real I/O latency (e.g., loading from S3/DynamoDB)
    return {
        "rule_1": "Always greet user",
        "rule_2": "Be concise",
        "loaded_at": datetime.now().isoformat(),
    }


@tool
def weather():
    """Get weather"""
    return "sunny"


@tool
def get_time():
    """Get current time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


model_id = "global.anthropic.claude-opus-4-5-20251101-v1:0"
model = BedrockModel(model_id=model_id)

agent = Agent(
    model=model,
    tools=[calculator, weather, get_time],
    system_prompt="""You're a helpful assistant. You can do simple math
    calculations, tell the weather, and provide the current time.
    Always start by acknowledging the user's name.
    When asked a follow-up like 'and that plus X', use the previous result from the conversation history.""",
)


def get_user_name(user_id):
    users = {"1": "Maira", "2": "Mani", "3": "Mark", "4": "Ishan", "5": "Dhawal"}
    return users.get(user_id, "Unknown")


@app.entrypoint
def strands_agent_bedrock_handling_context(payload, context):
    """AgentCore Runtime entrypoint with ping mode and conversation history."""
    global BIG_RULES, INVOCATION_COUNT, INIT_TIME

    INVOCATION_COUNT += 1
    request_start = time.perf_counter()

    # ── Cache loading (happens once per microVM) ──
    if BIG_RULES is None:
        INIT_TIME = datetime.now().isoformat()
        BIG_RULES = load_heavy_rules()
        cache_status = "COLD_START"
        cache_load_ms = 1200  # approximate
    else:
        cache_status = "WARM"
        cache_load_ms = 0

    mode = payload.get("mode", "chat")

    # ── PING MODE: No LLM, pure microVM overhead measurement ──
    if mode == "ping":
        elapsed_ms = (time.perf_counter() - request_start) * 1000
        return json.dumps({
            "pong": True,
            "cache_status": cache_status,
            "cache_loaded": BIG_RULES is not None,
            "invocation_count": INVOCATION_COUNT,
            "init_time": INIT_TIME,
            "session_id": context.session_id,
            "elapsed_ms": round(elapsed_ms, 2),
        })

    # ── CHAT MODE: LLM call with conversation history ──
    user_input = payload.get("prompt")
    user_id = payload.get("user_id", "1")
    user_name = get_user_name(user_id)

    # Build conversation history from RAM (only survives within same session/microVM)
    if user_id not in CONVERSATION_HISTORY:
        CONVERSATION_HISTORY[user_id] = []

    history = CONVERSATION_HISTORY[user_id]

    # Build prompt with history context
    history_text = ""
    if history:
        history_text = "\n\nPrevious conversation:\n"
        for role, msg in history[-10:]:  # last 10 exchanges
            history_text += f"  {role}: {msg}\n"

    prompt = f"""My name is {user_name}. Here is my request: {user_input}
    Additional context: This is session {context.session_id}.{history_text}
    Please acknowledge my name and provide assistance."""

    print(f"=== Invocation #{INVOCATION_COUNT} ===")
    print(f"Cache Status: {cache_status}")
    print(f"Session ID: {context.session_id}")
    print(f"User: {user_name} ({user_id})")
    print(f"History entries: {len(history)}")
    print(f"Prompt: {user_input}")

    response = agent(prompt)
    response_text = response.message['content'][0]['text']

    # Store in RAM history (persists only within this microVM session)
    history.append(("user", user_input))
    history.append(("assistant", response_text))

    return response_text


if __name__ == "__main__":
    app.run()
