import argparse
import gym
import os
import sys
import pickle
import time
import numpy as np
import torch
import math
from torch import nn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.policy import Policy
from models.value import Value
from models.policy_disc import DiscretePolicy
from models.discriminator import Discriminator
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch GAIL example')
    parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                        help='name of the environment to run')
    parser.add_argument('--expert-traj-path', metavar='G',
                        help='path of the expert trajectories')
    parser.add_argument('--render', action='store_true', default=False,
                        help='render the environment')
    parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                        help='log std for the policy (default: -0.0)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                        help='gae (default: 0.95)')
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                        help='l2 regularization regression (default: 1e-3)')
    parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                        help='learning rate (default: 3e-4)')
    parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                        help='clipping epsilon for PPO')
    parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                        help='number of threads for agent (default: 4)')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                        help='minimal batch size per PPO update (default: 2048)')
    parser.add_argument('--eval-batch-size', type=int, default=2048, metavar='N',
                        help='minimal batch size for evaluation (default: 2048)')
    parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                        help='maximal number of main iterations (default: 500)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                        help="interval between saving model (default: 0, means don't save)")
    parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
    return parser.parse_args()


def setup_environment(args):
    """Set up the environment, random seeds, and device"""
    # Set device
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    
    # Create environment
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    is_disc_action = len(env.action_space.shape) == 0
    action_dim = 1 if is_disc_action else env.action_space.shape[0]
    
    # Set up state normalization
    running_state = ZFilter((state_dim,), clip=5)
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    
    return env, state_dim, is_disc_action, action_dim, running_state, device, dtype


def setup_models(state_dim, action_dim, is_disc_action, env, args, device):
    """Create and initialize models and optimizers"""
    # Define actor and critic networks
    if is_disc_action:
        policy_net = DiscretePolicy(state_dim, env.action_space.n)
    else:
        policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std)
    
    value_net = Value(state_dim)
    discrim_net = Discriminator(state_dim + action_dim)
    discrim_criterion = nn.BCELoss()
    
    # Move models to device
    to_device(device, policy_net, value_net, discrim_net, discrim_criterion)
    
    # Setup optimizers
    optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)
    optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.learning_rate)
    
    return (policy_net, value_net, discrim_net, discrim_criterion, 
            optimizer_policy, optimizer_value, optimizer_discrim)


def load_expert_trajectories(args, running_state):
    """Load expert trajectories from file"""
    expert_traj, running_state = pickle.load(open(args.expert_traj_path, "rb"))
    running_state.fix = True
    return expert_traj, running_state


def create_reward_function(discrim_net, dtype):
    """Create custom reward function using the discriminator"""
    def expert_reward(state, action):
        state_action = tensor(np.hstack([state, action]), dtype=dtype)
        with torch.no_grad():
            return -math.log(discrim_net(state_action)[0].item())
    return expert_reward


def update_params(batch, i_iter, policy_net, value_net, discrim_net, discrim_criterion,
                 optimizer_policy, optimizer_value, optimizer_discrim,
                 expert_traj, args, device, dtype):
    """Update discriminator, policy, and value function parameters"""
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    
    with torch.no_grad():
        values = value_net(states)
        fixed_log_probs = policy_net.get_log_prob(states, actions)

    # Get advantage estimation from the trajectories
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

    # Update discriminator
    expert_state_actions = torch.from_numpy(expert_traj).to(dtype).to(device)
    g_o = discrim_net(torch.cat([states, actions], 1))
    e_o = discrim_net(expert_state_actions)
    optimizer_discrim.zero_grad()
    discrim_loss = discrim_criterion(g_o, ones((states.shape[0], 1), device=device)) + \
        discrim_criterion(e_o, zeros((expert_traj.shape[0], 1), device=device))
    discrim_loss.backward()
    optimizer_discrim.step()

    # Perform mini-batch PPO update
    optim_batch_size = 64
    optim_epochs = 10
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    
    for _ in range(optim_epochs):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm).to(device)

        states, actions, returns, advantages, fixed_log_probs = \
            states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()

        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
            states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

            ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, states_b, actions_b, returns_b,
                     advantages_b, fixed_log_probs_b, args.clip_epsilon, args.l2_reg)


def train(agent, expert_traj, policy_net, value_net, discrim_net, discrim_criterion,
         optimizer_policy, optimizer_value, optimizer_discrim,
         args, device, dtype):
    """Main training loop"""
    for i_iter in range(args.max_iter_num):
        # Generate multiple trajectories that reach the minimum batch_size
        discrim_net.to(torch.device('cpu'))
        batch, log = agent.collect_samples(args.min_batch_size, render=args.render)
        discrim_net.to(device)

        # Update parameters
        t0 = time.time()
        update_params(batch, i_iter, policy_net, value_net, discrim_net, discrim_criterion,
                     optimizer_policy, optimizer_value, optimizer_discrim,
                     expert_traj, args, device, dtype)
        t1 = time.time()
        
        # Evaluate with deterministic action (remove noise for exploration)
        discrim_net.to(torch.device('cpu'))
        _, log_eval = agent.collect_samples(args.eval_batch_size, mean_action=True)
        discrim_net.to(device)
        t2 = time.time()

        # Log training progress
        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\ttrain_discrim_R_avg {:.2f}\ttrain_R_avg {:.2f}\teval_discrim_R_avg {:.2f}\teval_R_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['avg_c_reward'], log['avg_reward'], 
                log_eval['avg_c_reward'], log_eval['avg_reward']))

        # Save the model
        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            to_device(torch.device('cpu'), policy_net, value_net, discrim_net)
            pickle.dump((policy_net, value_net, discrim_net), 
                       open(os.path.join(assets_dir(), 'learned_models/{}_gail.p'.format(args.env_name)), 'wb'))
            to_device(device, policy_net, value_net, discrim_net)

        # Clean up GPU memory
        torch.cuda.empty_cache()


def main():
    """Main function"""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set up environment and device
    env, state_dim, is_disc_action, action_dim, running_state, device, dtype = setup_environment(args)
    
    # Set up models and optimizers
    policy_net, value_net, discrim_net, discrim_criterion, optimizer_policy, optimizer_value, optimizer_discrim = \
        setup_models(state_dim, action_dim, is_disc_action, env, args, device)
    
    # Load expert trajectories
    expert_traj, running_state = load_expert_trajectories(args, running_state)
    
    # Create expert reward function
    expert_reward_func = create_reward_function(discrim_net, dtype)
    
    # Create agent
    agent = Agent(env, policy_net, device, custom_reward=expert_reward_func,
                 running_state=running_state, num_threads=args.num_threads)
    
    # Start training
    train(agent, expert_traj, policy_net, value_net, discrim_net, discrim_criterion,
         optimizer_policy, optimizer_value, optimizer_discrim,
         args, device, dtype)
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()
