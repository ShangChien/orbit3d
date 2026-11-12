"""PyTorch equivalents of the Uni-Core primitives used by the pinned architecture.

Registry decorators are retained only to preserve upstream source layout; Orbit3D
constructs explicit model configurations and never relies on the Uni-Core registry.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

BaseUnicoreModel = nn.Module
LayerNorm = nn.LayerNorm


def register_model(*args, **kwargs):
    return lambda cls: cls


def register_model_architecture(*args, **kwargs):
    return lambda function: function


def get_available_activation_fns():
    return ['gelu', 'relu', 'tanh', 'linear']


def get_activation_fn(name):
    return {'gelu': F.gelu, 'relu': F.relu, 'tanh': torch.tanh,
            'linear': lambda x: x}[name]


def permute_final_dims(tensor, inds):
    start = tensor.ndim-len(inds)
    return tensor.permute(list(range(start))+[start+i for i in inds])


def softmax_dropout(input, dropout_prob, is_training=True, mask=None, bias=None, inplace=True):
    x = input.contiguous()
    if not inplace:
        x = x.clone()
    if mask is not None:
        x = x + mask
    if bias is not None:
        x = x + bias
    return F.dropout(F.softmax(x, dim=-1), p=dropout_prob, training=is_training)


def init_bert_params(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0., std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0., std=.02)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()
