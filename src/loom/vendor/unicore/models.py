import torch.nn as nn

MODEL_REGISTRY = {}
ARCH_REGISTRY = {}

class BaseUnicoreModel(nn.Module):
    def __init__(self):
        super().__init__()

def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def register_model_architecture(model_name, arch_name):
    def decorator(fn):
        ARCH_REGISTRY[arch_name] = fn
        return fn
    return decorator

def build_model(args, task):
    arch_fn = ARCH_REGISTRY.get(getattr(args, 'arch', 'uni3dar'))
    if arch_fn is not None:
        arch_fn(args)
    model_cls = MODEL_REGISTRY[getattr(args, 'arch', 'uni3dar')]
    return model_cls.build_model(args, task)
