"""
Deploy the Memory Agent to AgentCore Runtime.
Creates an AgentCore Memory resource and deploys the memory-enabled agent.

Usage:
    python deploy_memory.py

Output:
    deploy_memory_info.json — contains agent_id, ecr_uri, region, memory_id
"""

import time
import json
from bedrock_agentcore_starter_toolkit import Runtime
from bedrock_agentcore.memory import MemoryClient
from boto3.session import Session

boto_session = Session()
region = boto_session.region_name
print(f"Deploying to region: {region}")

# ── Step 1: Create AgentCore Memory resource ──
print("\n--- Step 1: Creating AgentCore Memory resource ---")
memory_client = MemoryClient(region_name=region)
memory_name = "MemoryAgent"

try:
    memory = memory_client.create_memory_and_wait(
        name=memory_name,
        strategies=[],  # short-term only (conversation history)
        description="Durable conversation memory for microVM agent",
        event_expiry_days=90,
    )
    memory_id = memory['id']
    print(f"Memory created: {memory_id}")
except Exception as e:
    if "already exists" in str(e):
        print("Memory already exists, finding existing...")
        memories = memory_client.list_memories()
        memory_id = next(
            (m['id'] for m in memories if m['id'].startswith(memory_name)),
            None,
        )
        if memory_id:
            print(f"Using existing memory: {memory_id}")
        else:
            print(f"Could not find existing memory. Error: {e}")
            exit(1)
    else:
        print(f"Failed to create memory: {e}")
        exit(1)

# ── Step 2: Deploy agent to AgentCore Runtime ──
print("\n--- Step 2: Deploying Memory Agent to AgentCore Runtime ---")
agentcore_runtime = Runtime()
response = agentcore_runtime.configure(
    entrypoint="memory_agent.py",
    auto_create_execution_role=True,
    auto_create_ecr=True,
    requirements_file="requirements.txt",
    region=region,
    agent_name="memory_agent",
)
print("Configuration complete.")

launch_result = agentcore_runtime.launch()
print(f"Agent ID: {launch_result.agent_id}")
print(f"ECR URI: {launch_result.ecr_uri}")

# Poll for READY status
status_response = agentcore_runtime.status()
status = status_response.endpoint['status']
end_status = ['READY', 'CREATE_FAILED', 'DELETE_FAILED', 'UPDATE_FAILED']

print("Waiting for agent to be READY...")
while status not in end_status:
    time.sleep(10)
    status_response = agentcore_runtime.status()
    status = status_response.endpoint['status']
    print(f"  Status: {status}")

if status == 'READY':
    print("\nAgent is READY!")
    deploy_info = {
        "agent_id": launch_result.agent_id,
        "ecr_uri": launch_result.ecr_uri,
        "region": region,
        "memory_id": memory_id,
    }
    with open("deploy_memory_info.json", "w") as f:
        json.dump(deploy_info, f, indent=2)
    print(f"Deployment info saved to deploy_memory_info.json")
else:
    print(f"\nDeployment FAILED with status: {status}")
    exit(1)
