"""
Adapter module to allow using Gymnasium environments with code expecting Gym environments.
This handles the transition from Gym to Gymnasium API.
"""

import gymnasium as gym_new
import numpy as np

# Map old env names to new ones
ENV_MAPPING = {
    "Hopper-v2": "Hopper-v4",
    "HalfCheetah-v2": "HalfCheetah-v4",
    "Walker2d-v2": "Walker2d-v4",
    "Humanoid-v2": "Humanoid-v4",
    "Ant-v2": "Ant-v4",
    "Reacher-v2": "Reacher-v4",
    "Swimmer-v2": "Swimmer-v4",
    "InvertedPendulum-v2": "InvertedPendulum-v4",
    "InvertedDoublePendulum-v2": "InvertedDoublePendulum-v4"
}

class GymAdapter:
    """Adapter class to make Gymnasium environments work with old Gym code"""
    
    def __init__(self, env_name):
        """Create a Gymnasium environment with a Gym-like interface"""
        self.env_name = ENV_MAPPING.get(env_name, env_name)
        self.env = gym_new.make(self.env_name)
        
    def reset(self):
        """Adapt the reset method to return only the state"""
        state, _ = self.env.reset()
        return state
        
    def step(self, action):
        """Adapt the step method to match old Gym format"""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done, info
        
    def render(self):
        """Adapt render method"""
        return self.env.render()
    
    def close(self):
        """Close the environment"""
        return self.env.close()
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space

def make(env_name):
    """Drop-in replacement for gym.make"""
    return GymAdapter(env_name)
