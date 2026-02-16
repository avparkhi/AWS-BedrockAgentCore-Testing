"""
Deploy the agent to AgentCore Runtime.
Run this first before running tests.
"""

import time
import json
from bedrock_agentcore_starter_toolkit import Runtime
from boto3.session import Session

boto_session = Session()
region = boto_session.region_name
print(f"Deploying to region: {region}")

# Configure
agentcore_runtime = Runtime()
response = agentcore_runtime.configure(
    entrypoint="strands_claude_context.py",
    auto_create_execution_role=True,
    auto_create_ecr=True,
    requirements_file="requirements.txt",
    region=region,
    agent_name="strands_claude_context"
)
print("Configuration complete.")

# Launch
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
    # Save deployment info for tests
    deploy_info = {
        "agent_id": launch_result.agent_id,
        "ecr_uri": launch_result.ecr_uri,
        "region": region,
    }
    with open("deploy_info.json", "w") as f:
        json.dump(deploy_info, f, indent=2)
    print(f"Deployment info saved to deploy_info.json")
else:
    print(f"\nDeployment FAILED with status: {status}")
    exit(1)
