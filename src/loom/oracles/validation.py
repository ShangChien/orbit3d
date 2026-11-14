"""Separate checkpoint/scaler replay from repeated native-backbone numerical noise."""

import numpy as np
import torch


@torch.no_grad()
def compare_oracle_reload(original, restored, molecules, batch_size=16, runtime_atol=1e-5):
    original.adapter.eval()
    restored.adapter.eval()
    for key, value in original.adapter.state_dict().items():
        if not torch.equal(value, restored.adapter.state_dict()[key]):
            raise ValueError(f"Reloaded property parameter differs: {key}")
    groups = {}
    for i, mol in enumerate(molecules):
        original.validate_molecule(mol)
        restored.validate_molecule(mol)
        groups.setdefault(mol.GetNumAtoms(), []).append(i)
    predictions = np.empty((len(molecules), original.output_dim), dtype=np.float32)
    deviations = np.empty_like(predictions)
    max_shared = max_runtime = max_descriptor = 0.0
    for indices in groups.values():
        for start in range(0, len(indices), batch_size):
            ids = indices[start : start + batch_size]
            coordinates = torch.tensor(
                np.stack([molecules[i].GetConformer().GetPositions() for i in ids]),
                dtype=torch.float32,
                device=original.device,
            )
            symbols = [[a.GetSymbol() for a in molecules[i].GetAtoms()] for i in ids]
            features, _, _ = original.dpa_guidance.extract_native_batch(coordinates, symbols)
            prediction, std = original._properties_from_descriptor(features, coordinates)
            same, same_std = restored._properties_from_descriptor(features, coordinates)
            shared = max(float(abs(prediction - same).max()), float(abs(std - same_std).max()))
            if shared != 0.0:
                raise ValueError(f"Adapter/scaler reload differs on identical descriptors: {shared}")
            repeated, _, _ = restored.dpa_guidance.extract_native_batch(coordinates, symbols)
            again, again_std = restored._properties_from_descriptor(repeated, coordinates)
            runtime = max(float(abs(prediction - again).max()), float(abs(std - again_std).max()))
            if runtime > runtime_atol:
                raise ValueError(f"Repeated native prediction error {runtime} exceeds {runtime_atol}")
            max_shared = max(max_shared, shared)
            max_runtime = max(max_runtime, runtime)
            max_descriptor = max(max_descriptor, float(abs(features - repeated).max()))
            predictions[ids] = prediction.cpu().numpy()
            deviations[ids] = std.cpu().numpy()
    return (
        predictions,
        deviations,
        {
            "parameters_bitwise_equal": True,
            "same_descriptor_prediction_max_abs_diff": max_shared,
            "repeated_native_prediction_max_abs_diff_eV": max_runtime,
            "repeated_descriptor_max_abs_diff": max_descriptor,
            "runtime_atol_eV": runtime_atol,
        },
    )
