"""Select the requested forcefield consistently for octree and property training."""

from .dpa_forcefield import DPAPotentialGuidance
from .hyff_forcefield import HyFFPotentialGuidance


def build_forcefield(backend, weights, atom_types, device, *, tier='vivace', route='reference'):
    if backend == 'dpa4':
        return DPAPotentialGuidance(weights, atom_type_map=atom_types, device=device)
    if backend == 'hyff':
        return HyFFPotentialGuidance(tier=tier, weights=weights, atom_type_map=atom_types,
                                     device=device, route=route)
    raise ValueError(f'Unknown forcefield backend: {backend}')
