# AgentCore MicroVM - Streaming Agent with Hybrid Memory

A production-ready conversational agent running on **AWS Bedrock AgentCore Runtime** with:

- **Hybrid memory** - RAM (fast, same session) + Durable (persistent, cross-session)
- **Real-time streaming** - tokens appear as they're generated (~1.5s to first token vs ~5.4s without)
- **Multi-user isolation** - each user has their own memory context
- **12-test automated suite** - ping, chat, memory persistence, user isolation

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   AgentCore Runtime                  │
│                                                      │
│  ┌──────────────┐    ┌──────────────────────────┐   │
│  │   microVM     │    │    AgentCore Memory API   │   │
│  │              │    │    (durable, cross-session)│   │
│  │  ┌────────┐  │    └──────────────────────────┘   │
│  │  │ Model  │  │               ▲                    │
│  │  │(global)│  │               │ save/load          │
│  │  └────────┘  │               │                    │
│  │  ┌────────┐  │    ┌──────────┴───────────┐       │
│  │  │ Agent  │──┼───▶│  MemoryHookProvider   │       │
│  │  │(per-   │  │    │  - on_initialized:    │       │
│  │  │ call)  │  │    │    load history       │       │
│  │  └────────┘  │    │  - on_message_added:  │       │
│  │  ┌────────┐  │    │    save to durable    │       │
│  │  │  RAM   │  │    └──────────────────────┘       │
│  │  │history │  │                                    │
│  │  └────────┘  │                                    │
│  └──────────────┘                                    │
└─────────────────────────────────────────────────────┘
```

**Warm start:** RAM has context → no API call (~0ms overhead)
**Cold start:** RAM empty → loads from durable memory (~300ms)

## Prerequisites

- **Python 3.11+** (the SDK requires it)
- **AWS account** with Bedrock AgentCore access
- **AWS credentials** configured (`aws configure` or environment variables)
- **No Docker required** — CodeBuild handles container builds in the cloud

## Quick Start

### 1. Clone and setup

```bash
git clone <this-repo>
cd agentcore_microvm

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Deploy

```bash
.venv/bin/python deploy_memory.py
```

This will:
- Create an AgentCore Memory resource (or reuse existing)
- Build an ARM64 container via CodeBuild (~40s)
- Deploy to AgentCore Runtime
- Save `deploy_memory_info.json` with agent_id, memory_id, region

### 3. Run tests

```bash
.venv/bin/python test_memory_agent.py
```

Runs 12 tests covering ping (cold/warm), chat (memory, continuity), persistence (cross-session), and isolation (per-user, per-session). Expected: 12/12 pass.

### 4. Chat

```bash
# Non-streaming (full response at once)
.venv/bin/python chat_memory.py

# Streaming (tokens appear in real-time)
.venv/bin/python chat_memory_streaming.py
```

### 5. Cleanup (when done)

```bash
.venv/bin/python cleanup_memory.py
```

Deletes the AgentCore runtime, Memory resource, and ECR repository.

## Project Files

### Agent Code

| File | Description |
|------|-------------|
| `memory_agent.py` | Main agent — hybrid memory, streaming, tools (weather, time, calculator) |
| `strands_claude_context.py` | Simpler agent — RAM-only memory, no streaming (v1) |

### Deploy / Cleanup

| File | Description |
|------|-------------|
| `deploy_memory.py` | Deploy memory agent (creates Memory resource + Runtime) |
| `deploy.py` | Deploy basic agent (no memory, v1) |
| `cleanup_memory.py` | Delete all memory agent resources |
| `cleanup.py` | Delete all basic agent resources |
| `requirements.txt` | Python dependencies |

### Chat Clients

| File | Description |
|------|-------------|
| `chat_memory.py` | Interactive chat — non-streaming, uses SDK `runtime.invoke()` |
| `chat_memory_streaming.py` | Interactive chat — streaming via boto3 direct, shows TTFT |
| `chat.py` | Chat client for the basic agent (v1) |

### Tests

| File | Description |
|------|-------------|
| `test_memory_agent.py` | 12-test suite — ping, chat, memory, isolation |
| `test_microvm_latency.py` | Latency benchmarks for the basic agent |
| `test_quick_latency.py` | Quick latency check |

## Chat Commands

Both `chat_memory.py` and `chat_memory_streaming.py` support:

| Command | What it does |
|---------|--------------|
| `/ping` | Ping microVM — shows cold/warm status, no LLM call |
| `/new` | New runtime session (triggers cold start, tests durable memory) |
| `/user <1-5>` | Switch user (1=Maira, 2=Mani, 3=Mark, 4=Ishan, 5=Dhawal) |
| `/memory` | Show last 5 turns from durable memory |
| `/forget` | Clear memory (new memory session) |
| `/session` | Show current runtime session ID |
| `/sessions` | List all sessions used this run |
| `/switch <id>` | Switch to a previous session |
| `/msession` | Show memory session info |
| `/latency` | Show latency log for all requests |
| `/clear` | Clear screen |
| `/quit` | Exit |

## Agent Modes

The agent entrypoint handles three modes via the `mode` payload field:

| Mode | LLM Call | Description |
|------|----------|-------------|
| `chat` (default) | Yes | Conversational AI with hybrid memory |
| `ping` | No | Returns cache status, invocation count — isolates microVM overhead |
| `memory_query` | No | Returns stored turns from durable memory |

## Streaming

Streaming is controlled by the `stream` payload field (default: `true`).

**How it works on the agent side:**
- The entrypoint dispatches to a generator (streaming) or a regular function (non-streaming)
- The SDK auto-detects generators and wraps them as SSE (`text/event-stream`)
- Text chunks are yielded as they arrive from `agent.stream_async()`

**How it works on the client side:**
- The SDK's `runtime.invoke()` returns `{}` for streaming — no programmatic access
- `chat_memory_streaming.py` uses boto3 `invoke_agent_runtime()` directly to parse SSE chunks
- First chunk = metadata (memory source, cache status), middle = text tokens, last = done

**Performance:**

| Metric | Non-Streaming | Streaming |
|--------|---------------|-----------|
| Time to first token | ~5.4s | ~1–1.5s |
| Total response time | ~5.4s | ~5.4s |

## Key Gotchas

These are the hard-won lessons not covered in official docs.

### 1. `payload` is not a regular dict

It's a `JSONSerializableDict`. `.get("key", default)` with 2 args throws TypeError. Use `.get("key") or "default"` instead.

### 2. SDK double-encodes return values

If you return `json.dumps({"key": "val"})`, the client gets `"{\"key\": \"val\"}"` (a string, not a dict). Use `<<META:key=val>>` tag format for non-streaming metadata.

### 3. Streaming generators: yield raw objects, never json.dumps()

The SDK JSON-encodes each yielded value. If you `yield json.dumps(my_dict)`, the SDK encodes it again → client gets a string instead of a dict. Always yield raw Python dicts/strings.

### 4. Non-streaming responses truncate at 1024 bytes

No error, no warning. Keep payloads compact.

### 5. Use venv Python, not system Python

macOS system Python is 3.9.x. The SDK needs 3.11+. Always use `.venv/bin/python`.

## Latency Reference

Measured from 12-test runs:

| Operation | Cold Start | Warm |
|-----------|-----------|------|
| Ping (no LLM) | ~1.7s | ~180ms |
| Chat (LLM + memory) | ~8–17s | ~5.4s |
| Chat streaming (TTFT) | — | ~1–1.5s |
| Memory query (no LLM) | — | ~350ms |

## IAM Permissions

The runtime role (auto-created by deploy) needs these for observability:

**CloudWatch Logs:**
- `logs:CreateLogGroup`, `logs:CreateLogStream`, `logs:PutLogEvents`
- `logs:DescribeLogStreams`, `logs:DescribeLogGroups`
- Resource: `arn:aws:logs:REGION:ACCOUNT:log-group:/aws/bedrock-agentcore/runtimes/*`

**X-Ray:**
- `xray:PutTraceSegments`, `xray:PutTelemetryRecords`
- `xray:GetSamplingRules`, `xray:GetSamplingTargets`

**CloudWatch Metrics:**
- `cloudwatch:PutMetricData` (namespace: `bedrock-agentcore`)

## License

MIT
