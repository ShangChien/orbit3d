"""Paired property metrics in physical units, with sample correspondence preserved."""
import numpy as np


def paired_energy_difference_metrics(predictions, reference, target_names, first, second):
    """Inputs use eV; compare each molecule's first-minus-second energy difference."""
    predicted, actual = np.asarray(predictions), np.asarray(reference)
    if predicted.ndim != 2 or predicted.shape != actual.shape or not len(predicted):
        raise ValueError('Energy difference metrics require equally shaped, nonempty paired samples')
    if predicted.shape[1] != len(target_names) or len(set(target_names)) != len(target_names):
        raise ValueError('Energy difference target order is ambiguous')
    if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ValueError('Nonfinite energy pairs must be recorded as failures, not averaged into metrics')
    left, right = target_names.index(first), target_names.index(second)
    error = (predicted[:, left]-predicted[:, right])-(actual[:, left]-actual[:, right])
    return {'n': len(error), 'difference': f'{first} - {second}', 'unit': 'eV',
            'mae_eV': float(np.abs(error).mean()), 'rmse_eV': float(np.sqrt(np.square(error).mean())),
            'mean_signed_error_eV': float(error.mean()),
            'absolute_difference_of_means_eV': float(abs(error.mean()))}
