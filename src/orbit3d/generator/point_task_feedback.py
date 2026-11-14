"""Raw-coordinate property feedback without a force-field dependency."""
import numpy as np
from .molecular_state import mol3d_to_validated_rdkit_mol
from ..oracles.unimol2_task import support_exclusion


def evaluate_point_task(instances, task, points, reward_type='inverse_point_distance'):
    records, molecules, indices = {}, [], []
    for i, instance in instances:
        mol = mol3d_to_validated_rdkit_mol(instance)
        excluded = support_exclusion(mol)
        if excluded:
            records[i] = dict(evaluation_status='not_supported', reason=excluded)
        else:
            molecules.append(mol)
            indices.append(i)
    predictions = task.predict(molecules)
    for i, values in zip(indices, predictions):
        if not np.isfinite(values).all():
            raise FloatingPointError('Nonfinite frozen point-task prediction')
        distance = points.distance(values)
        records[i] = dict(evaluation_status='completed', prediction=values.tolist(),
            point_distance=distance, property_hit=points.intervals().hit(values),
            reward=1. if reward_type=='eligibility_only' else 1./(1.+distance))
    return records
