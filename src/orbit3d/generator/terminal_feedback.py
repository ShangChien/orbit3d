"""Separate structural, conditional physical and terminal property comparisons."""
import torch


def terminal_advantages(structural, physical, property_eligible, quality, level):
    """All channels update the original sampled trajectory; no relaxed trajectory ratios.

    Physical comparisons condition on structurally admitted proposals. Property
    comparisons condition on accepted, evaluable endpoints. A conditional group
    with fewer than two members has no within-group comparison signal.
    """
    if level not in {'structure', 'physical', 'property'}:
        raise ValueError('Unknown terminal feedback level')
    if bool((physical & ~structural).any()) or bool((property_eligible & ~physical).any()):
        raise ValueError('Terminal eligibility must be nested')
    n=len(quality)
    structure=structural.to(quality.dtype)
    structure=(structure-structure.mean())/structure.std(unbiased=False).clamp_min(1e-6)
    def conditional(values, mask):
        out=torch.zeros_like(quality);count=int(mask.sum())
        if count>=2:
            selected=values[mask]
            out[mask]=(n/count)*(selected-selected.mean())/selected.std(unbiased=False).clamp_min(.1)
        return out
    physics=conditional(physical.to(quality.dtype),structural) if level!='structure' else torch.zeros_like(quality)
    prop=conditional(quality,property_eligible) if level=='property' else torch.zeros_like(quality)
    total=structure+physics+prop;normalizer=total.std(unbiased=False).clamp_min(1.)
    return total/normalizer,{'structure':structure/normalizer,'physical':physics/normalizer,'property':prop/normalizer}
