import numpy as np

class ZFilter:
    """
    State normalization utility
    """
    def __init__(self, shape, center=True, scale=True, clip=10.0, epsilon=1e-8):
        self.shape = shape
        self.center = center
        self.scale = scale
        self.clip = clip
        self.epsilon = epsilon
        self.rs = None
        self.running_mean = np.zeros(shape, dtype=np.float64)
        self.running_variance = np.ones(shape, dtype=np.float64)
        self.count = 0
        self.fix = False
        
    def __call__(self, x):
        if self.fix:
            return self.normalize_fixed(x)
        else:
            return self.normalize_update(x)
            
    def normalize_update(self, x):
        self.count += 1
        self.running_mean = self.running_mean + (x - self.running_mean) / self.count
        self.running_variance = self.running_variance + (x - self.running_mean) * (x - self.running_mean) / self.count
        
        return self.normalize_fixed(x)
        
    def normalize_fixed(self, x):
        x_normalized = x.copy()
        
        if self.center:
            x_normalized = x_normalized - self.running_mean
            
        if self.scale:
            x_normalized = x_normalized / (np.sqrt(self.running_variance) + self.epsilon)
            
        if self.clip:
            x_normalized = np.clip(x_normalized, -self.clip, self.clip)
            
        return x_normalized