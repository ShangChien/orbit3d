"""One frozen predictor contract for B-defined labels, rewards and acceptance."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem

from .unimol2_oracle import InternalUniMol2Oracle, source_features
from ..generator.molecular_state import mol_to_clean_smiles


B_ELEMENTS = frozenset(['H', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'I'])
B_WEIGHT_SHA256 = '108b50b8d9910144cd840446b86b2f95500bec5c27cf8a08de20b5cc2701d385'


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def support_exclusion(mol):
    if mol.GetNumAtoms() > 80:
        return 'more_than_80_total_atoms'
    unseen = sorted({a.GetSymbol() for a in mol.GetAtoms()} - B_ELEMENTS)
    if unseen:
        return 'unsupported_elements:' + ','.join(unseen)
    try:
        source_features(mol)
    except ValueError as error:
        return str(error)
    return None


class FrozenUniMol2Task:
    """No force, uncertainty or coordinate-gradient interface is implied."""
    def __init__(self, directory, selection, device='cuda:0'):
        directory = Path(directory)
        selected = json.loads(Path(selection).read_text())
        hashes = {name: file_sha256(directory/name) for name in ['model.pt', 'oracle_meta.json']}
        if hashes != {name: selected['hashes'][selected['selected_unimol']][name] for name in hashes}:
            raise ValueError('B differs from the frozen development-selected predictor')
        if hashes['model.pt'] != B_WEIGHT_SHA256:
            raise ValueError('This task requires the selected 1.1B checkpoint')
        meta = json.loads((directory/'oracle_meta.json').read_text())
        if meta['model_size'] != '1.1B' or meta['target_cols'] != ['homo', 'lumo']:
            raise ValueError('Wrong B architecture or target order')
        self.contract = dict(version=1, hashes=hashes, mean=meta['mean'], scale=meta['scale'],
                             target_cols=meta['target_cols'], target_units=meta['target_units'],
                             input_contract=meta['input_contract'], precision='float32', mode='eval',
                             supported_elements=sorted(B_ELEMENTS), max_total_atoms=80,
                             role='Same fixed B defines training labels, rewards and success; not independent DFT truth',
                             preprocessing_sha256=file_sha256(Path(__file__).with_name('unimol2_oracle.py')))
        self.oracle = InternalUniMol2Oracle.load_from_dir(directory, device)
        self.oracle.model.float().eval().requires_grad_(False)
        self.molecule_calls = 0
        self.forward_calls = 0
        self.oracle.model.register_forward_hook(self._count_forward)

    def _count_forward(self, module, args, result):
        self.forward_calls += 1

    def predict(self, mols, batch_size=16):
        for mol in mols:
            error = support_exclusion(mol)
            if error:
                raise ValueError(error)
        with torch.autocast(device_type=torch.device(self.oracle.device).type, enabled=False):
            predictions, _ = self.oracle.predict_3d_mols(mols, batch_size=batch_size)
        self.molecule_calls += len(mols)
        return predictions


def graph_set(sdf):
    molecules = list(Chem.SDMolSupplier(str(sdf), removeHs=False))
    if any(m is None for m in molecules):
        raise ValueError(f'Invalid reference SDF: {sdf}')
    return {mol_to_clean_smiles(m) for m in molecules}


def summarize_task(rows, target, goal, source_graphs, asset_graphs, tolerance=.25):
    """Only rescore when explicitly requested; never create new proposal counts."""
    column = ['homo', 'lumo'].index(target)
    evaluated = [r for r in rows if r.get('evaluated')]
    hit_rows = [r for r in evaluated if abs(r['prediction_eV'][column]-goal) <= tolerance]
    graphs = {r['smiles'] for r in hit_rows}
    errors = np.array([r['prediction_eV'][column]-goal for r in evaluated])
    n = len(rows)
    return dict(attempted=n, legal=sum(r['legal'] for r in rows), evaluated=len(evaluated),
                unsupported=sum(r.get('support_exclusion') is not None for r in rows),
                hits=len(hit_rows), hit_yield=len(hit_rows)/n if n else None,
                conditional_hit_rate=len(hit_rows)/len(evaluated) if evaluated else None,
                mae_eV=float(np.abs(errors).mean()) if evaluated else None,
                bias_eV=float(errors.mean()) if evaluated else None,
                prediction_mean_eV=float(errors.mean()+goal) if evaluated else None,
                prediction_std_eV=float(errors.std()) if evaluated else None,
                unique_hit_graphs=len(graphs), unique_source_novel_hits=len(graphs-source_graphs),
                unique_all_asset_novel_hits=len(graphs-asset_graphs),
                hit_tolerances={str(t):sum(abs(r['prediction_eV'][column]-goal)<=t for r in evaluated) for t in [.1,.25,.5]})
