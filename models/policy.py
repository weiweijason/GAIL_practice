import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)  # Wider network
        self.fc2 = nn.Linear(128, 64)
        self.action_head = nn.Linear(64, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action_head(x), dim=-1)
        return action_probs
        
    def act(self, state):
        """Returns action and log probability for state"""
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def evaluate(self, states, actions):
        """Returns log probs, entropy for batch of states, actions"""
        action_probs = self.forward(states)
        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return action_log_probs, entropy
    
    def get_log_prob(self, states, actions):
        """Returns log probs for states, actions"""
        action_probs = self.forward(states)
        dist = Categorical(action_probs)
        return dist.log_prob(actions)