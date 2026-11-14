"""Explicit class weighting estimated only from the supervised training structures."""
from collections import Counter
import math
import torch


def inverse_sqrt_element_weights(molecules, dictionary, device='cpu'):
    counts = Counter(atom.GetSymbol() for mol in molecules for atom in mol.GetAtoms())
    if not counts:
        raise ValueError('Element weighting requires nonempty training structures')
    largest = max(counts.values())
    values = {symbol: math.sqrt(largest/count) for symbol, count in sorted(counts.items())}
    weights = torch.ones(len(dictionary), dtype=torch.float32, device=device)
    for symbol, value in values.items():
        weights[dictionary[symbol]] = value
    return weights, {'scheme': 'inverse_sqrt_training_atom_frequency', 'training_atom_counts': dict(counts),
                     'element_weights': values, 'special_token_weight': 1.,
                     'reduction': 'Weighted atom CE divided by the sum of observed-label weights; tree and XYZ unchanged'}
