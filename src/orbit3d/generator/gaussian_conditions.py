"""Mean/width request sampling and reciprocal condition preferences."""
import math
import torch


def sample_gaussian_conditions(target, scale, active, sigma_range=(.1, .5), dropout=.1):
    low, high = sigma_range
    if not 0 < low <= high:
        raise ValueError('Preference widths must be positive and ordered')
    # One physical width per batch; pairing does not mix narrow/wide preferences.
    sigma_eV = torch.empty((), device=target.device).uniform_(math.log(low), math.log(high)).exp()
    sigma = (sigma_eV / torch.as_tensor(scale, device=target.device, dtype=target.dtype)).expand_as(target)
    mu = target + sigma * torch.randn_like(target)
    keep = torch.rand((len(target), 1), device=target.device) >= dropout
    mask = (active[None, :] & keep).unsqueeze(-1).expand(-1, -1, 2)
    return mu, sigma, mask


def reciprocal_pairs(target, mu, sigma, mask, minimum_log_gap=.5):
    active = mask.any(-1)
    # Current experiment has one active property, but compatibility sums active axes.
    column = int(active.any(0).to(torch.int64).argmax())
    order = target[:, column].argsort()
    partner = torch.empty_like(order)
    partner[order] = order.flip(0)
    own = (-.5 * ((target-mu)/sigma).square() - sigma.log()).masked_fill(~active, 0).sum(-1)
    cross = (-.5 * ((target-mu[partner])/sigma[partner]).square() - sigma[partner].log()).masked_fill(~active, 0).sum(-1)
    gap = own - cross
    eligible = active.any(-1) & active[partner].any(-1) & (gap >= minimum_log_gap) & (gap[partner] >= minimum_log_gap)
    return partner, eligible
