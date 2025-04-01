import torch
import torch.nn as nn
from torch.distributions import Categorical

class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_num):
        super(DiscretePolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_num)
        
        self.action_num = action_num

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        action_probs = torch.softmax(self.fc3(x), dim=1)
        return action_probs
    
    def select_action(self, x):
        action_probs = self.forward(x)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action
    
    def get_kl(self, x):
        action_probs = self.forward(x)
        return action_probs * torch.log(action_probs.mean(dim=0, keepdim=True) / action_probs)
    
    def get_log_prob(self, x, actions):
        action_probs = self.forward(x)
        dist = Categorical(action_probs)
        return dist.log_prob(actions.long().flatten()).unsqueeze(1)
    
    def get_fim(self, x):
        action_probs = self.forward(x)
        M = action_probs.size(0)
        
        # Calculate Fisher Information Matrix
        fim = torch.zeros(M, self.action_num, self.action_num)
        for i in range(M):
            p = action_probs[i]
            fim[i] = torch.diag(p) - torch.outer(p, p)
            
        return fim