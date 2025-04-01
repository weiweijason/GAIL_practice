import torch
import numpy as np

def estimate_advantages(rewards, masks, values, gamma, tau, device):
    """
    Compute advantage estimates using Generalized Advantage Estimation (GAE)
    """
    rewards = torch.Tensor(rewards).to(device)
    masks = torch.Tensor(masks).to(device)
    values = torch.Tensor(values).to(device)
    
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1).to(device)
    advantages = tensor_type(rewards.size(0), 1).to(device)
    
    prev_value = 0
    prev_advantage = 0
    
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]
        
        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]
        
    returns = values + advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns

def conjugate_gradient(Avp, b, nsteps, residual_tol=1e-10):
    """
    Conjugate gradient algorithm to efficiently compute the inverse Fisher-vector product
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    
    for i in range(nsteps):
        Avp_p = Avp(p)
        alpha = rdotr / torch.dot(p, Avp_p)
        x += alpha * p
        r -= alpha * Avp_p
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        
    return x