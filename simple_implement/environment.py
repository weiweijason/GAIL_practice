import numpy as np
import gym
from gym import spaces

class CustomerServiceEnv(gym.Env):
    def __init__(self):
        # State space: customer query embedding (simplified as 10-dim vector)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,))
        
        # Action space: response type (10 different response categories)
        self.action_space = spaces.Discrete(10)
        
        # Current customer query
        self.current_state = None
        
        # Database of sample customer queries
        self.query_database = self._load_query_database()
        
    def _load_query_database(self):
        # In real implementation, load from actual database
        # Here we simulate with random embeddings
        return np.random.randn(100, 10)
        
    def reset(self):
        # Select random customer query as initial state
        self.current_state = self.query_database[np.random.randint(0, len(self.query_database))]
        return self.current_state
        
    def step(self, action):
        # In real system, this would use actual customer satisfaction metrics
        # Here we use a simplified reward based on action-state compatibility
        next_state_idx = np.random.randint(0, len(self.query_database))
        next_state = self.query_database[next_state_idx]
        
        # For GAIL, this reward isn't directly used for training
        # The discriminator will generate the rewards
        reward = 0
        
        # Episode ends after each interaction in this simplified version
        done = True
        
        self.current_state = next_state
        return next_state, reward, done, {}