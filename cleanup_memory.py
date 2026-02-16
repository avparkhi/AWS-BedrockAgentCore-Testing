"""
Cleanup Memory Agent resources.
Deletes the AgentCore Memory resource, runtime agent, and ECR repository.
"""

import json
import boto3
from bedrock_agentcore.memory import MemoryClient

with open("deploy_memory_info.json") as f:
    info = json.load(f)

region = info["region"]
agent_id = info["agent_id"]
ecr_uri = info["ecr_uri"]
memory_id = info.get("memory_id")

print(f"Cleaning up memory agent: {agent_id}")
print(f"Region: {region}")
if memory_id:
    print(f"Memory ID: {memory_id}")

# ── Delete AgentCore Memory resource ──
if memory_id:
    print("\n--- Deleting AgentCore Memory resource ---")
    try:
        memory_client = MemoryClient(region_name=region)
        memory_client.delete_memory_and_wait(memory_id=memory_id)
        print(f"Memory deleted: {memory_id}")
    except Exception as e:
        print(f"Failed to delete memory: {e}")

# ── Delete AgentCore Runtime ──
print("\n--- Deleting AgentCore Runtime ---")
agentcore_control_client = boto3.client('bedrock-agentcore-control', region_name=region)
try:
    agentcore_control_client.delete_agent_runtime(agentRuntimeId=agent_id)
    print(f"Agent runtime deleted: {agent_id}")
except Exception as e:
    print(f"Failed to delete agent runtime: {e}")

# ── Delete ECR repository ──
print("\n--- Deleting ECR repository ---")
ecr_client = boto3.client('ecr', region_name=region)
try:
    repo_name = ecr_uri.split('/')[1]
    ecr_client.delete_repository(repositoryName=repo_name, force=True)
    print(f"ECR repository deleted: {repo_name}")
except Exception as e:
    print(f"Failed to delete ECR repo: {e}")

print("\nCleanup complete.")
