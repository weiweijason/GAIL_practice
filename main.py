import argparse
import gym
import os
import pickle
import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from models.policy import Policy
from models.value import Value
from models.discriminator import Discriminator
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent
from utils.zfilter import ZFilter
from utils.torch_utils import to_device, tensor, ones, zeros

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch GAIL implementation')
    parser.add_argument('--env-name', default="Hopper-v3", metavar='G',
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
    parser.add_argument('--seed', type=int, default=42, metavar='N',
                        help='random seed (default: 42)')
    parser.add_argument('--min-batch-size', type=int, default