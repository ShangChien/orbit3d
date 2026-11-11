import torch
import torch.nn as nn
import torch.nn.functional as F

class _Utils:
    @staticmethod
    def get_activation_fn(activation: str):
        if activation == 'relu':
            return F.relu
        elif activation == 'gelu':
            return F.gelu
        elif activation == 'tanh':
            return torch.tanh
        elif activation in ('silu', 'swish'):
            return F.silu
        elif activation == 'linear':
            return lambda x: x
        raise ValueError(f'Unsupported activation: {activation}')

utils = _Utils()

class _Metrics:
    @staticmethod
    def log_scalar(*args, **kwargs):
        pass

metrics = _Metrics()
