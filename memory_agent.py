"""
AgentCore Runtime Agent with Hybrid Memory (Hooks-based)
========================================================
Uses Strands hooks pattern for clean memory integration:
  - AgentInitializedEvent: Loads history (RAM or durable) into system prompt
  - MessageAddedEvent: Saves each message to durable memory automatically

Hybrid memory:
  - RAM history: Fast, same microVM session (no API call on warm starts)
  - Durable memory: Persists across restarts, redeploys, session changes

Modes:
  - "ping": No LLM call. Returns cache + memory status.
  - "chat": LLM call with hybrid memory via hooks.
  - "memory_query": Return stored turns from durable memory (no LLM).
"""

import time
import json
import os
import asyncio
import queue
import threading
from strands import Agent, tool
from strands.hooks import AgentInitializedEvent, HookProvider, HookRegistry, MessageAddedEvent
from strands_tools import calculator
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands.models import BedrockModel
from bedrock_agentcore.memory import MemoryClient
from datetime import datetime

app = BedrockAgentCoreApp()

# ── Globals that persist within a microVM session ──
BIG_RULES = None
CONVERSATION_HISTORY = {}  # keyed by user_id, stores list of (role, message) tuples
INVOCATION_COUNT = 0
INIT_TIME = None

# ── Model (global, expensive to init — reused across invocations) ──
model_id = "global.anthropic.claude-opus-4-5-20251101-v1:0"
model = BedrockModel(model_id=model_id)

# ── Memory client (lazy-initialized, reused within microVM session) ──
MEMORY_CLIENT = None


def get_memory_client():
    """Lazy-init memory client (reused within microVM session)."""
    global MEMORY_CLIENT
    if MEMORY_CLIENT is None:
        region = os.environ.get("AWS_REGION", "us-east-1")
        MEMORY_CLIENT = MemoryClient(region_name=region)
    return MEMORY_CLIENT


def load_heavy_rules():
    """Simulate loading expensive rules/config."""
    time.sleep(1.2)
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


def get_user_name(user_id):
    users = {"1": "Maira", "2": "Mani", "3": "Mark", "4": "Ishan", "5": "Dhawal"}
    return users.get(user_id, "Unknown")


def _parse_turn_messages(turns):
    """Format durable memory turns into context lines."""
    lines = []
    for turn in turns:
        for message in turn:
            role = message.get("role") or "?"
            content_obj = message.get("content") or {}
            text = content_obj.get("text") or "" if hasattr(content_obj, 'get') else ""
            lines.append(f"  {role}: {text}")
    return lines


# ── Memory Hook Provider ──

class MemoryHookProvider(HookProvider):
    """Hooks-based memory integration for Strands Agent.

    - on_agent_initialized: Loads conversation history into system prompt
      (RAM on warm starts, durable memory on cold starts)
    - on_message_added: Saves each message to durable memory automatically

    Tracks memory_source on the instance (not agent.state) to avoid
    JSONSerializableDict limitations.
    """

    def __init__(self, memory_client, memory_id, actor_id, session_id, ram_history):
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.actor_id = actor_id
        self.session_id = session_id
        self.ram_history = ram_history
        self.memory_source = "none"  # tracked here, not in agent.state

    def on_agent_initialized(self, event: AgentInitializedEvent):
        """Load conversation history when agent is created (each invocation)."""
        try:
            if self.ram_history:
                # WARM: RAM has context from current microVM session (no API call)
                context_lines = [f"  {role}: {msg}" for role, msg in self.ram_history[-10:]]
                context = "\n".join(context_lines)
                event.agent.system_prompt += (
                    f"\n\nConversation history (current session, from RAM):\n{context}"
                )
                self.memory_source = "ram"

                # Also check durable memory for older context from previous sessions
                if self.memory_id:
                    turns = self.memory_client.get_last_k_turns(
                        memory_id=self.memory_id,
                        actor_id=self.actor_id,
                        session_id=self.session_id,
                        k=5,
                    )
                    if turns and len(turns) > len(self.ram_history) // 2:
                        durable_lines = _parse_turn_messages(turns)
                        durable_context = "\n".join(durable_lines)
                        event.agent.system_prompt += (
                            f"\n\nOlder history (previous sessions, from durable memory):\n{durable_context}"
                        )
                        self.memory_source = "both"

            elif self.memory_id:
                # COLD START: RAM is empty, load from durable memory
                turns = self.memory_client.get_last_k_turns(
                    memory_id=self.memory_id,
                    actor_id=self.actor_id,
                    session_id=self.session_id,
                    k=5,
                )
                if turns:
                    context_lines = _parse_turn_messages(turns)
                    context = "\n".join(context_lines)
                    event.agent.system_prompt += (
                        f"\n\nConversation history (from durable memory — survives across sessions):\n{context}"
                    )
                    self.memory_source = "durable"
                    print(f"[MEMORY] Loaded {len(turns)} turns from durable memory")

        except Exception as e:
            print(f"[MEMORY] Hook load error: {e}")

    def on_message_added(self, event: MessageAddedEvent):
        """Save each message to durable memory automatically."""
        if not self.memory_id:
            return

        try:
            messages = event.agent.messages
            if not messages:
                return

            last_msg = messages[-1]

            # Extract text from the message content
            last_content = last_msg.get("content") if hasattr(last_msg, 'get') else None
            if not last_content:
                return

            text = None
            if isinstance(last_content, list) and last_content:
                first = last_content[0]
                if isinstance(first, dict):
                    text = first.get("text")
                elif hasattr(first, 'get'):
                    text = first.get("text")
            elif isinstance(last_content, str):
                text = last_content

            if text:
                role_val = last_msg.get("role") if hasattr(last_msg, 'get') else "user"
                role = (role_val or "user").upper()
                self.memory_client.create_event(
                    memory_id=self.memory_id,
                    actor_id=self.actor_id,
                    session_id=self.session_id,
                    messages=[(text, role)],
                )
        except Exception as e:
            print(f"[MEMORY] Hook save error: {e}")

    def register_hooks(self, registry: HookRegistry):
        registry.add_callback(AgentInitializedEvent, self.on_agent_initialized)
        registry.add_callback(MessageAddedEvent, self.on_message_added)


def create_agent_with_memory(memory_id, actor_id, memory_session_id, ram_history):
    """Create agent per-invocation with memory hooks.

    Model is global (reused). Agent creation is lightweight (no network calls).
    Returns (agent, hook_provider) so caller can read hook_provider.memory_source.
    """
    hook = None
    hooks = []
    if memory_id:
        memory_client = get_memory_client()
        hook = MemoryHookProvider(memory_client, memory_id, actor_id, memory_session_id, ram_history)
        hooks = [hook]

    agent = Agent(
        model=model,
        tools=[calculator, weather, get_time],
        system_prompt="""You're a helpful assistant with durable memory across sessions.
    You can do simple math calculations, tell the weather, and provide the current time.
    Always start by acknowledging the user's name.
    When asked a follow-up like 'and that plus X', use the previous result from the conversation history.
    If conversation history is provided, use it to maintain context — even across different sessions.""",
        hooks=hooks,
    )
    return agent, hook


def _run_async_stream(agent, prompt, chunk_queue):
    """Worker thread: runs async streaming generator, pushes text chunks to queue."""
    async def _consume():
        async for event in agent.stream_async(prompt):
            if isinstance(event, dict) and "data" in event:
                chunk_queue.put(event["data"])

    try:
        asyncio.run(_consume())
    except Exception as e:
        chunk_queue.put(("__ERROR__", str(e)))
    finally:
        chunk_queue.put(None)  # sentinel


def _streaming_handler(payload, context):
    """Generator entrypoint — yields chunks for SSE streaming."""
    global BIG_RULES, INVOCATION_COUNT, INIT_TIME

    INVOCATION_COUNT += 1
    request_start = time.perf_counter()

    # ── Cache loading (happens once per microVM) ──
    if BIG_RULES is None:
        INIT_TIME = datetime.now().isoformat()
        BIG_RULES = load_heavy_rules()
        cache_status = "COLD_START"
    else:
        cache_status = "WARM"

    mode = payload.get("mode") or "chat"

    # ── PING MODE: single yield ──
    # Yield raw dicts — SDK handles JSON encoding for SSE.
    # Using json.dumps() here would cause double-encoding (SDK gotcha).
    if mode == "ping":
        elapsed_ms = (time.perf_counter() - request_start) * 1000
        yield {
            "pong": True,
            "cache_status": cache_status,
            "cache_loaded": BIG_RULES is not None,
            "invocation_count": INVOCATION_COUNT,
            "init_time": INIT_TIME,
            "session_id": context.session_id,
            "elapsed_ms": round(elapsed_ms, 2),
            "memory_enabled": True,
        }
        return

    # ── MEMORY QUERY MODE: single yield ──
    if mode == "memory_query":
        memory_id = payload.get("memory_id")
        actor_id = payload.get("user_id") or "1"
        memory_session_id = payload.get("memory_session_id") or "default"
        k = payload.get("k") or 5

        try:
            client = get_memory_client()
            turns = client.get_last_k_turns(
                memory_id=memory_id,
                actor_id=actor_id,
                session_id=memory_session_id,
                k=k,
            )
            compact_turns = []
            for turn in (turns or []):
                compact_turn = []
                for msg in turn:
                    content = msg.get("content") or {} if hasattr(msg, 'get') else {}
                    text = content.get("text") or "" if hasattr(content, 'get') else ""
                    role = msg.get("role") or "?" if hasattr(msg, 'get') else "?"
                    compact_turn.append({"role": role, "content": {"text": text[:80]}})
                compact_turns.append(compact_turn)
            yield {"turns": compact_turns, "count": len(turns) if turns else 0}
        except Exception as e:
            yield {"error": str(e), "turns": [], "count": 0}
        return

    # ── CHAT MODE: streaming ──
    user_input = payload.get("prompt")
    user_id = payload.get("user_id") or "1"
    user_name = get_user_name(user_id)
    memory_id = payload.get("memory_id")
    memory_session_id = payload.get("memory_session_id") or "default"

    if user_id not in CONVERSATION_HISTORY:
        CONVERSATION_HISTORY[user_id] = []
    ram_history = CONVERSATION_HISTORY[user_id]

    agent, hook = create_agent_with_memory(memory_id, user_id, memory_session_id, ram_history)
    memory_source = hook.memory_source if hook else "none"

    prompt = f"""My name is {user_name}. Here is my request: {user_input}
    Additional context: This is runtime session {context.session_id}.
    Please acknowledge my name and provide assistance."""

    print(f"=== Invocation #{INVOCATION_COUNT} (streaming) ===")
    print(f"Cache Status: {cache_status}")
    print(f"Runtime Session: {context.session_id}")
    print(f"Memory Session: {memory_session_id}")
    print(f"User: {user_name} ({user_id})")
    print(f"RAM history entries: {len(ram_history)}")
    print(f"Memory source: {memory_source}")
    print(f"Prompt: {user_input}")

    # Yield meta (first chunk) — raw dict, SDK handles JSON encoding
    yield {
        "type": "meta",
        "memory_source": memory_source,
        "cache_status": cache_status,
        "session_id": context.session_id,
    }

    # Stream via thread + queue bridge (async → sync)
    chunk_queue = queue.Queue()
    worker = threading.Thread(target=_run_async_stream, args=(agent, prompt, chunk_queue))
    worker.start()

    full_text = []
    while True:
        chunk = chunk_queue.get()
        if chunk is None:
            break
        if isinstance(chunk, tuple) and chunk[0] == "__ERROR__":
            yield {"type": "error", "message": chunk[1]}
            break
        full_text.append(str(chunk))
        yield str(chunk)

    worker.join()

    # Save to RAM (persists within this microVM session)
    response_text = "".join(full_text)
    ram_history.append(("user", user_input))
    ram_history.append(("assistant", response_text))

    # Yield done (final chunk) — raw dict, SDK handles JSON encoding
    memory_source = hook.memory_source if hook else "none"
    yield {
        "type": "done",
        "memory_source": memory_source,
        "ram_entries": len(ram_history),
    }


def _non_streaming_handler(payload, context):
    """Non-streaming entrypoint — returns string (backward compatible)."""
    global BIG_RULES, INVOCATION_COUNT, INIT_TIME

    INVOCATION_COUNT += 1
    request_start = time.perf_counter()

    # ── Cache loading (happens once per microVM) ──
    if BIG_RULES is None:
        INIT_TIME = datetime.now().isoformat()
        BIG_RULES = load_heavy_rules()
        cache_status = "COLD_START"
    else:
        cache_status = "WARM"

    mode = payload.get("mode") or "chat"

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
            "memory_enabled": True,
        })

    # ── MEMORY QUERY MODE: Return stored turns, no LLM ──
    if mode == "memory_query":
        memory_id = payload.get("memory_id")
        actor_id = payload.get("user_id") or "1"
        memory_session_id = payload.get("memory_session_id") or "default"
        k = payload.get("k") or 5

        try:
            client = get_memory_client()
            turns = client.get_last_k_turns(
                memory_id=memory_id,
                actor_id=actor_id,
                session_id=memory_session_id,
                k=k,
            )
            # Truncate turn text to fit within SDK 1024-byte response limit
            compact_turns = []
            for turn in (turns or []):
                compact_turn = []
                for msg in turn:
                    content = msg.get("content") or {} if hasattr(msg, 'get') else {}
                    text = content.get("text") or "" if hasattr(content, 'get') else ""
                    role = msg.get("role") or "?" if hasattr(msg, 'get') else "?"
                    compact_turn.append({"role": role, "content": {"text": text[:80]}})
                compact_turns.append(compact_turn)
            return json.dumps({"turns": compact_turns, "count": len(turns) if turns else 0})
        except Exception as e:
            return json.dumps({"error": str(e), "turns": [], "count": 0})

    # ── CHAT MODE: Hooks-based hybrid memory ──
    user_input = payload.get("prompt")
    user_id = payload.get("user_id") or "1"
    user_name = get_user_name(user_id)
    memory_id = payload.get("memory_id")
    memory_session_id = payload.get("memory_session_id") or "default"

    # Initialize RAM history for this user if needed
    if user_id not in CONVERSATION_HISTORY:
        CONVERSATION_HISTORY[user_id] = []
    ram_history = CONVERSATION_HISTORY[user_id]

    # Create agent per-invocation (lightweight — model is global)
    # Hooks fire on creation: load memory into system prompt
    agent, hook = create_agent_with_memory(memory_id, user_id, memory_session_id, ram_history)
    memory_source = hook.memory_source if hook else "none"

    prompt = f"""My name is {user_name}. Here is my request: {user_input}
    Additional context: This is runtime session {context.session_id}.
    Please acknowledge my name and provide assistance."""

    print(f"=== Invocation #{INVOCATION_COUNT} ===")
    print(f"Cache Status: {cache_status}")
    print(f"Runtime Session: {context.session_id}")
    print(f"Memory Session: {memory_session_id}")
    print(f"User: {user_name} ({user_id})")
    print(f"RAM history entries: {len(ram_history)}")
    print(f"Memory source: {memory_source}")
    print(f"Prompt: {user_input}")

    # Call agent — on_message_added hook saves to durable memory automatically
    response = agent(prompt)
    response_text = response.message['content'][0]['text']

    # Save to RAM (persists within this microVM session)
    ram_history.append(("user", user_input))
    ram_history.append(("assistant", response_text))

    # Return response with memory metadata as simple delimited format
    # (Avoids double-JSON-encoding by SDK)
    memory_source = hook.memory_source if hook else "none"
    meta = f"<<META:memory_source={memory_source},cache_status={cache_status},ram_entries={len(ram_history)},session_id={context.session_id}>>"
    return f"{meta}\n{response_text}"


@app.entrypoint
def memory_agent_entrypoint(payload, context):
    """AgentCore Runtime entrypoint with hooks-based hybrid memory.

    Dispatches to streaming (generator) or non-streaming (string return)
    based on the 'stream' payload flag (default: True).
    """
    stream = payload.get("stream")
    if stream is None:
        stream = True
    if stream:
        return _streaming_handler(payload, context)
    else:
        return _non_streaming_handler(payload, context)


if __name__ == "__main__":
    app.run()
