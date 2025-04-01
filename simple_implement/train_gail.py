import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from environment import CustomerServiceEnv
from policy_network import Policy
from discriminator import Discriminator

# Initialize environment
env = CustomerServiceEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize networks
policy = Policy(state_dim, action_dim)
discriminator = Discriminator(state_dim, action_dim)

# Optimizers
policy_optimizer = optim.Adam(policy.parameters(), lr=0.001)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

# Load expert demonstrations
# In a real system, these would be logs of human agent actions
def load_expert_demonstrations(n_demos=50):
    # Simulate expert demonstrations with random states and expert policies
    expert_states = []
    expert_actions = []
    
    for _ in range(n_demos):
        state = env.reset()
        # Expert policy (in real system, this would be human agent's action)
        # Here we simulate with rule-based logic
        action = np.argmax(np.dot(state, np.random.randn(state_dim, action_dim)))
        expert_states.append(state)
        expert_actions.append(action)
    
    return np.array(expert_states), np.array(expert_actions)

expert_states, expert_actions = load_expert_demonstrations()

# GAIL training loop
def train_gail(n_epochs=1000, n_steps_per_epoch=10):
    for epoch in tqdm(range(n_epochs)):
        # Sample trajectories using current policy
        states, actions, log_probs = sample_trajectories(n_steps_per_epoch)
        
        # Update discriminator
        discriminator_loss = update_discriminator(states, actions, expert_states, expert_actions)
        
        # Calculate rewards using the discriminator
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.LongTensor(actions)
            # Expert is labeled as 1, policy as 0
            # So we take -log(D) as the reward
            rewards = -torch.log(1 - discriminator(states_tensor, actions_tensor)).squeeze().detach().numpy()
        
        # Update policy (simple policy gradient)
        policy_loss = update_policy(states, actions, log_probs, rewards)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Discriminator loss = {discriminator_loss:.4f}, Policy loss = {policy_loss:.4f}")

def sample_trajectories(n_steps):
    states = []
    actions = []
    log_probs = []
    
    for _ in range(n_steps):
        state = env.reset()
        action, log_prob = policy.act(state)
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
    
    return np.array(states), np.array(actions), torch.stack(log_probs)

def update_discriminator(policy_states, policy_actions, expert_states, expert_actions):
    # Convert to tensors
    policy_states = torch.FloatTensor(policy_states)
    policy_actions = torch.LongTensor(policy_actions)
    expert_states = torch.FloatTensor(expert_states)
    expert_actions = torch.LongTensor(expert_actions)
    
    # Expert demos labeled as 1, policy samples as 0
    policy_labels = torch.zeros(len(policy_states))
    expert_labels = torch.ones(len(expert_states))
    
    # Train discriminator
    discriminator_optimizer.zero_grad()
    
    # Discriminator predictions
    policy_preds = discriminator(policy_states, policy_actions).squeeze()
    expert_preds = discriminator(expert_states, expert_actions).squeeze()
    
    # Binary classification loss
    policy_loss = F.binary_cross_entropy(policy_preds, policy_labels)
    expert_loss = F.binary_cross_entropy(expert_preds, expert_labels)
    discriminator_loss = policy_loss + expert_loss
    
    discriminator_loss.backward()
    discriminator_optimizer.step()
    
    return discriminator_loss.item()

def update_policy(states, actions, old_log_probs, rewards):
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    
    # Simple policy gradient update
    policy_optimizer.zero_grad()
    policy_loss = -(old_log_probs * torch.FloatTensor(rewards)).mean()
    policy_loss.backward()
    policy_optimizer.step()
    
    return policy_loss.item()

# Run training
train_gail()

# Save the trained policy
torch.save(policy.state_dict(), "customer_service_policy.pt")
