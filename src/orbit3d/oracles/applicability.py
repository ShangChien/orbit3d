"""Fingerprint applicability checks against the property model's actual training set."""

import json
from pathlib import Path
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from .graph_features import fingerprint_graph, NORMALIZED_GRAPH, RAW_GRAPH


class MorganDomain:
    def __init__(self, reference_smiles, cutoff, graph_normalization=NORMALIZED_GRAPH):
        self.reference_smiles = list(reference_smiles)
        self.cutoff = float(cutoff)
        self.graph_normalization = graph_normalization
        self.generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
        molecules = [fingerprint_graph(Chem.MolFromSmiles(s), self.graph_normalization) for s in self.reference_smiles]
        self.references = [self.generator.GetFingerprint(m) for m in molecules]
        self.training_elements = {a.GetSymbol() for m in molecules for a in Chem.AddHs(m).GetAtoms()}
        if not self.references or not 0 <= self.cutoff <= 1:
            raise ValueError('Applicability domain requires reference structures and a similarity cutoff')

    def score(self, molecule):
        fp = self.generator.GetFingerprint(fingerprint_graph(molecule, self.graph_normalization))
        return max(DataStructs.BulkTanimotoSimilarity(fp, self.references))

    def assess(self, molecule):
        similarity = self.score(molecule)
        unseen = sorted({a.GetSymbol() for a in molecule.GetAtoms()} - self.training_elements)
        return {'similarity': similarity, 'unseen_training_elements': unseen,
                'in_domain': not unseen and similarity >= self.cutoff}

    def save(self, path):
        Path(path).write_text(json.dumps({'schema_version':2,'graph_normalization':self.graph_normalization,
                                         'reference_smiles': self.reference_smiles, 'cutoff': self.cutoff,
                                         'fingerprint': 'Morgan radius2 2048 bits chiral',
                                         'criterion': 'training-element coverage and Morgan cutoff',
                                         'training_elements': sorted(self.training_elements)}, indent=2)+'\n')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        version=data.get('schema_version',1)
        if version not in {1,2}:raise ValueError('Unknown applicability-domain schema')
        return cls(data['reference_smiles'], data['cutoff'],
                   RAW_GRAPH if version==1 else data['graph_normalization'])
