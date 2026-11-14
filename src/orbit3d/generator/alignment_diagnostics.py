"""Optional, synchronized stage timings and read-only policy gradient probes."""
import time
import torch


class StageTimer:
    def __init__(self, device, enabled):
        self.device, self.enabled = device, enabled
        self.seconds = {}
        self.started = {}

    def start(self, name):
        if self.enabled:
            if self.device.type == 'cuda':torch.cuda.synchronize(self.device)
            self.started[name] = time.perf_counter()

    def stop(self, name):
        if self.enabled:
            if self.device.type == 'cuda':torch.cuda.synchronize(self.device)
            self.seconds[name] = self.seconds.get(name, 0.) + time.perf_counter()-self.started.pop(name)


def categorical_kl_loss(current, reference, template):
    """Match the per-molecule/per-decision normalization of the training objective."""
    total=torch.zeros_like(template);counts=torch.zeros_like(template)
    for (logp,_,owner),(ref,_,_) in zip(current,reference):
        delta=torch.where(torch.isfinite(logp),logp-ref,torch.zeros_like(logp))
        total.index_add_(0,owner,(logp.exp()*delta).sum(-1))
        counts.index_add_(0,owner,torch.ones_like(owner,dtype=template.dtype))
    return (total/counts.clamp_min(1)).mean()


def gradient_probe(terms, parameters):
    """autograd.grad does not populate .grad or update the optimizer/RNG state."""
    gradients={}
    for name,loss in terms.items():
        values=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
        gradients[name]=[torch.zeros_like(p,dtype=torch.float64) if g is None else g.detach().double()
                         for p,g in zip(parameters,values)]
    dot=lambda a,b:sum((x*y).sum() for x,y in zip(a,b))
    norms={name:float(dot(g,g).sqrt()) for name,g in gradients.items()}
    result={'weighted_gradient_norms':norms,'policy_cosines':{}}
    for name in ['kl','replay']:
        if name in gradients:
            denominator=norms['policy']*norms[name]
            result['policy_cosines'][name]=float(dot(gradients['policy'],gradients[name]))/denominator if denominator else None
    return result
