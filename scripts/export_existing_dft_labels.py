"""Export a DFT-supervised view of the admitted, dual-labelled generated dataset."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem

from orbit3d.utils.dataset_io import write_precise_sdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        label_fields={'homo/lumo': 'DFT_homo/DFT_lumo',
                      'B_homo/B_lumo': 'Unchanged frozen predictor values'},
        geometry='Original quantum input; MLIP terminal, not DFT-optimized',
        split='Inherits the admitted dual-label dataset; no new split or sample selection',
        new_DFT=False, files={})
    for split in ['train', 'valid', 'test']:
        source = args.dataset / f'{split}_generated.sdf'
        molecules = list(Chem.SDMolSupplier(str(source), removeHs=False))
        for mol in molecules:
            if mol is None:
                raise ValueError(f'Invalid SDF record in {source}')
            for name in ['homo', 'lumo']:
                value = mol.GetProp('DFT_' + name)
                if not np.isfinite(float(value)):
                    raise ValueError(f'Nonfinite DFT {name} in {source}')
                mol.SetProp(name, value)
            mol.SetProp('LabelTheory', mol.GetProp('DFTProtocol'))
            mol.SetProp('PropertyLabelOrigin', 'existing_DFT_singlepoint')
        output = args.output / f'{split}.sdf'
        write_precise_sdf(output, molecules)
        restored = list(Chem.SDMolSupplier(str(output), removeHs=False))
        assert len(restored) == len(molecules)
        for before, after in zip(molecules, restored):
            np.testing.assert_array_equal(before.GetConformer().GetPositions(),
                                          after.GetConformer().GetPositions())
            assert Chem.MolToSmiles(before) == Chem.MolToSmiles(after)
            for name in before.GetPropNames():
                assert before.GetProp(name) == after.GetProp(name)
        report['files'][output.name] = dict(
            molecules=len(molecules),
            sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            input_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
