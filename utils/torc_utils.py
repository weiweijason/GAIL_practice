import numpy as np
import torch

def to_device(device, *args):
    """
    Move tensors to the specified device
    """
    return [x.to(device) for x in args]

def tensor(x, dtype=None):
    """
    Convert to tensor with optional dtype
    """
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    x = np.asarray(x)
    x = torch.from_numpy(x)
    if dtype is not None:
        x = x.type(dtype)
    return x

def ones(*shape, device=None):
    """
    Return ones tensor with specified shape
    """
    return torch.ones(*shape, device=device)

def zeros(*shape, device=None):
    """
    Return zeros tensor with specified shape
    """
    return torch.zeros(*shape, device=device)