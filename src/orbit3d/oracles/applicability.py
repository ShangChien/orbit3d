"""Fingerprint applicability checks against the property model's actual training set."""

import json
from pathlib import Path
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


class MorganDomain:
    def __init__(self, reference_smiles, cutoff):
        self.reference_smiles = list(reference_smiles)
        self.cutoff = float(cutoff)
        self.generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
        self.references = [self.generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in self.reference_smiles]
        if not self.references or not 0 <= self.cutoff <= 1:
            raise ValueError('Applicability domain requires reference structures and a similarity cutoff')

    def score(self, molecule):
        fp = self.generator.GetFingerprint(Chem.RemoveHs(molecule))
        return max(DataStructs.BulkTanimotoSimilarity(fp, self.references))

    def save(self, path):
        Path(path).write_text(json.dumps({'reference_smiles': self.reference_smiles, 'cutoff': self.cutoff,
                                         'fingerprint': 'Morgan radius2 2048 bits chiral'}, indent=2)+'\n')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        return cls(data['reference_smiles'], data['cutoff'])
