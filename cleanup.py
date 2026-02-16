"""
Cleanup deployed AgentCore resources.
Deletes the runtime agent and ECR repository.
"""

import json
import boto3

with open("deploy_info.json") as f:
    info = json.load(f)

region = info["region"]
agent_id = info["agent_id"]
ecr_uri = info["ecr_uri"]

print(f"Cleaning up agent: {agent_id}")
print(f"Region: {region}")

# Delete AgentCore runtime
agentcore_control_client = boto3.client('bedrock-agentcore-control', region_name=region)
try:
    runtime_delete_response = agentcore_control_client.delete_agent_runtime(
        agentRuntimeId=agent_id,
    )
    print(f"Agent runtime deleted: {agent_id}")
except Exception as e:
    print(f"Failed to delete agent runtime: {e}")

# Delete ECR repository
ecr_client = boto3.client('ecr', region_name=region)
try:
    repo_name = ecr_uri.split('/')[1]
    response = ecr_client.delete_repository(
        repositoryName=repo_name,
        force=True
    )
    print(f"ECR repository deleted: {repo_name}")
except Exception as e:
    print(f"Failed to delete ECR repo: {e}")

print("Cleanup complete.")
