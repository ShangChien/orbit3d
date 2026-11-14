TASK_REGISTRY = {}

class UnicoreTask:
    def __init__(self, args):
        self.args = args
        self.datasets = {}

def register_task(name):
    def decorator(cls):
        TASK_REGISTRY[name] = cls
        return cls
    return decorator
