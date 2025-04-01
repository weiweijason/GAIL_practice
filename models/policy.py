import torch

def ppo_step(policy_net, value_net, policy_optimizer, value_optimizer, optim_value_iternum, 
             states, actions, returns, advantages, fixed_log_probs, clip_epsilon, l2_reg):
    """
    Proximal Policy Optimization update step
    """
    # PPO value network update
    for _ in range(optim_value_iternum):
        values_pred = value_net(states)
        value_loss = (values_pred - returns).pow(2).mean()
        
        # Weight decay for regularization
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
            
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()
    
    # PPO policy network update
    log_probs = policy_net.get_log_prob(states, actions)
    ratio = torch.exp(log_probs - fixed_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Weight decay for regularization
    for param in policy_net.parameters():
        policy_loss += param.pow(2).sum() * l2_reg
        
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()
    
    # For logging
    mean_kl = 0.5 * ((fixed_log_probs - log_probs) ** 2).mean()
    mean_entropy = -log_probs.mean()
    
    return value_loss.item(), policy_loss.item(), mean_kl.item(), mean_entropy.item()