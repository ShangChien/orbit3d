"""Point-condition preference losses on normalized generation scores, not posteriors."""
import copy
import torch
from torch.nn import functional as F


def nearby_size_matched_pairs(values, atoms, enabled, minimum=.25, maximum=1., atom_gap=8):
    """Disjoint near-window pairs; no change to the sampled training batch."""
    n=len(values);partner=torch.arange(n,device=values.device);eligible=torch.zeros(n,dtype=torch.bool,device=values.device)
    edges=[]
    for i in range(n):
        for j in range(i+1,n):
            distance=float(abs(values[i]-values[j]))
            if bool(enabled[i] and enabled[j]) and minimum<=distance<=maximum and abs(int(atoms[i])-int(atoms[j]))<=atom_gap:
                edges.append((distance,i,j))
    for _,i,j in sorted(edges):
        if not bool(eligible[i] or eligible[j]):
            partner[i]=j;partner[j]=i;eligible[i]=eligible[j]=True
    return partner,eligible


def bidirectional_preference_loss(positive, negative, reference_positive, reference_negative, partner, eligible, margin=.02):
    """Both condition swaps and molecule swaps, relative to the common frozen start."""
    if not bool(eligible.any()):return positive.sum()*0,{}
    own=reference_positive-positive;cross=reference_negative-negative
    condition_delta=own[eligible]-cross[eligible]
    sample_delta=own[eligible]-cross[partner][eligible]
    condition=F.relu(margin-condition_delta).mean();sample=F.relu(margin-sample_delta).mean()
    return (condition+sample)/2,dict(condition_loss=float(condition.detach()),sample_loss=float(sample.detach()),
        condition_preference=float((condition_delta>0).float().mean()),sample_preference=float((sample_delta>0).float().mean()))


def bidirectional_softmax_loss(positive, negative, reference_positive, reference_negative, partner, eligible, temperature=.02):
    """Two-way cross entropy on both axes of each 2x2 preference pair.

    A hinge is linear until the margin is reached. In that region, averaging
    over reciprocal pairs makes its two axes algebraically identical. The
    smooth cross entropy retains score-dependent weights on both axes.
    """
    if not bool(eligible.any()):return positive.sum()*0,{}
    own=reference_positive-positive;cross=reference_negative-negative
    condition_delta=own[eligible]-cross[eligible]
    sample_delta=own[eligible]-cross[partner][eligible]
    condition=temperature*F.softplus(-condition_delta/temperature).mean()
    sample=temperature*F.softplus(-sample_delta/temperature).mean()
    return (condition+sample)/2,dict(condition_loss=float(condition.detach()),sample_loss=float(sample.detach()),
        condition_preference=float((condition_delta>0).float().mean()),sample_preference=float((sample_delta>0).float().mean()))


def nearest_condition_candidates(values, k=4):
    if not 2<=k<=len(values):raise ValueError('K must fit the training batch')
    distance=(values[:,None]-values[None]).abs();distance.fill_diagonal_(float('inf'))
    others=distance.argsort(dim=1,stable=True)[:,:k-1]
    return torch.cat([torch.arange(len(values),device=values.device)[:,None],others],dim=1)


def multi_condition_loss(losses, reference_losses, distances_eV, enabled, tolerance=.25, temperature=.1):
    """Soft distance labels; the generation score is not a calibrated likelihood."""
    if not bool(enabled.any()):return losses.sum()*0,{}
    logits=(reference_losses-losses)/temperature
    targets=F.softmax(-distances_eV/tolerance,dim=1)
    # Multiplying by temperature keeps score gradients bounded as temperature varies.
    loss=temperature*(-targets*F.log_softmax(logits,dim=1)).sum(1)[enabled].mean()
    return loss,dict(soft_target_entropy=float((-(targets*targets.clamp_min(1e-12).log()).sum(1))[enabled].mean()),
                     matched_condition_top1=float((logits.argmax(1)==0)[enabled].float().mean()))


class PointPreferenceObjective:
    """Frozen-start score differences with identical dropout-free scoring passes."""
    def __init__(self, model, method, property_index, scale):
        if method not in {'bidirectional', 'bidirectional_softmax', 'multi_condition'}:
            raise ValueError('Unknown point preference method')
        self.reference = copy.deepcopy(model).eval().requires_grad_(False)
        self.method, self.property_index, self.scale = method, property_index, scale

    def __call__(self, model, packed, target, lower, upper, mask, atoms):
        index = self.property_index
        values = target[:, index] * self.scale[index]
        enabled = mask[:, index].all(-1)
        was_training = model.training
        model.eval()

        def scores(condition):
            lo, hi = lower[condition], upper[condition]
            # Keep each row's condition-dropout mask, not the other molecule's.
            def run(net):
                return net.compute_sft_and_understanding_loss(
                    packed, target, lo, hi, mask, understanding_weight=0,
                    return_per_molecule=True)[1]['per_molecule_generation_loss']
            current = run(model)
            with torch.no_grad():
                reference = run(self.reference)
            return current, reference

        try:
            if self.method in {'bidirectional', 'bidirectional_softmax'}:
                partner, eligible = nearby_size_matched_pairs(values, atoms, enabled)
                if not bool(eligible.any()):
                    return next(model.interval_conditioner.parameters()).sum()*0, {'preference_eligible': 0}
                positive, ref_positive = scores(torch.arange(len(values), device=values.device))
                negative, ref_negative = scores(partner)
                objective = bidirectional_softmax_loss if self.method == 'bidirectional_softmax' else bidirectional_preference_loss
                loss, metrics = objective(positive, negative, ref_positive, ref_negative, partner, eligible)
                metrics['preference_eligible'] = int(eligible.sum())
            else:
                candidates = nearest_condition_candidates(values, k=4)
                pairs = [scores(candidates[:, k]) for k in range(4)]
                current = torch.stack([p[0] for p in pairs], dim=1)
                reference = torch.stack([p[1] for p in pairs], dim=1)
                distance = (values[:, None]-values[candidates]).abs()
                loss, metrics = multi_condition_loss(current, reference, distance, enabled)
                metrics['preference_eligible'] = int(enabled.sum())
        finally:
            model.train(was_training)
        return loss, metrics
