"""Equal-width property sampling with bounded per-structure exposure."""
import numpy as np


def _allocate(weights, mass, cap):
    if len(weights)*cap < mass-1e-12:
        raise ValueError('Requested source/replay mass exceeds per-structure exposure cap')
    low, high = 0., mass/weights.min()
    for _ in range(70):
        scale=(low+high)/2
        if np.minimum(scale*weights,cap).sum()<mass:low=scale
        else:high=scale
    return np.minimum(high*weights,cap)


def property_sampling_probabilities(values, bins=20, max_exposure=5., source_count=None, max_generated_fraction=.25):
    values=np.asarray(values,dtype=float)
    if values.ndim!=1 or len(values)==0 or not np.isfinite(values).all() or bins<1 or max_exposure<1:
        raise ValueError('Property sampling requires finite scalar labels and exposure >= 1')
    edges=np.linspace(values.min(),values.max(),bins+1)
    ids=np.clip(np.searchsorted(edges,values,side='right')-1,0,bins-1)
    counts=np.bincount(ids,minlength=bins);weights=1/counts[ids]
    cap=max_exposure/len(values);probabilities=_allocate(weights,1.,cap)
    if source_count is not None:
        if not 0<source_count<len(values) or not 0<max_generated_fraction<1:
            raise ValueError('Replay requires nonempty source and generated subsets')
        generated_mass=min(float(probabilities[source_count:].sum()),max_generated_fraction)
        probabilities=np.concatenate([_allocate(weights[:source_count],1-generated_mass,cap),
                                      _allocate(weights[source_count:],generated_mass,cap)])
    probabilities/=probabilities.sum()
    return probabilities,dict(scheme='equal_width_capped',edges_eV=edges.tolist(),bin_counts=counts.tolist(),
        bins=bins,max_exposure=max_exposure,max_observed_relative_probability=float(probabilities.max()*len(values)),
        generated_fraction=float(probabilities[source_count:].sum()) if source_count is not None else 0.)
