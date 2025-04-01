import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        # Input: state and action
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, state, action):
        # For discrete actions, convert to one-hot
        if action.dim() == 1:
            action = F.one_hot(action.long(), num_classes=10).float()
        
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output between 0 and 1: probability of being from expert
        x = torch.sigmoid(self.fc3(x))
        return x