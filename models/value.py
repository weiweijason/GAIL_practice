import torch
import torch.nn as nn
import torch.nn.functional as F

class Value(nn.Module):
    def __init__(self, state_dim):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.value_head = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)
        return value