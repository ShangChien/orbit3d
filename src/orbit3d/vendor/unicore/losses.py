import torch.nn as nn

LOSS_REGISTRY = {}

class UnicoreLoss(nn.Module):
    def __init__(self, task):
        super().__init__()
        self.task = task

def register_loss(name):
    def decorator(cls):
        LOSS_REGISTRY[name] = cls
        return cls
    return decorator
