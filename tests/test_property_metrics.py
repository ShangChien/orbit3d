import numpy as np
import pytest
from orbit3d.utils.property_metrics import paired_energy_difference_metrics


def test_difference_of_ensemble_means_is_not_a_paired_gap_mae():
    prediction = np.array([[3., 1.], [1., 3.]])
    reference = np.full((2, 2), 2.)
    result = paired_energy_difference_metrics(prediction, reference, ['s0s1','s0t1'], 's0s1', 's0t1')
    assert result['mae_eV'] == 2. and result['rmse_eV'] == 2.
    assert result['absolute_difference_of_means_eV'] == 0.
    with pytest.raises(ValueError, match='paired samples'):
        paired_energy_difference_metrics(prediction, reference[:1], ['s0s1','s0t1'], 's0s1','s0t1')
    prediction[0, 0] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        paired_energy_difference_metrics(prediction, reference, ['s0s1','s0t1'], 's0s1','s0t1')
