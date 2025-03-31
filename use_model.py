import torch
import numpy as np
from environment import CustomerServiceEnv
from policy_network import Policy

# Initialize environment
env = CustomerServiceEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Load trained policy
policy = Policy(state_dim, action_dim)
policy.load_state_dict(torch.load("customer_service_policy.pt"))

# Define response mapping (would be more detailed in real implementation)
response_types = [
    "Greeting", "Account query", "Billing question", 
    "Technical support", "Product information", "Complaint handling",
    "Refund request", "Service upgrade", "Cancellation", "General information"
]

def generate_response(query_embedding, action):
    """Generate appropriate response based on query and action type"""
    # In a real system, this would use NLG or templates to generate actual responses
    return f"Response using template type: {response_types[action]}"

# Test the trained agent
def test_agent(n_interactions=5):
    for i in range(n_interactions):
        # Get a customer query
        state = env.reset()
        print(f"\nCustomer Query {i+1}: (embedding visualization) {state[:3]}...")
        
        # Get action from policy
        action, _ = policy.act(state)
        
        # Generate response
        response = generate_response(state, action)
        print(f"Bot Action: Type {action} - {response_types[action]}")
        print(f"Response: {response}")

test_agent()