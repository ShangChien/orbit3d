"""Versioned graph preparation for fingerprints; source geometry stays untouched."""
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

RAW_GRAPH = 'rdkit_raw_v1'
NORMALIZED_GRAPH = 'rdkit_normalize_v1'


def fingerprint_graph(molecule, normalization=NORMALIZED_GRAPH):
    if molecule is None:
        raise ValueError('Invalid molecule for graph features')
    graph=Chem.RemoveHs(Chem.Mol(molecule))
    Chem.SanitizeMol(graph)
    if normalization == NORMALIZED_GRAPH:
        return rdMolStandardize.Normalize(graph)
    if normalization == RAW_GRAPH:
        return graph
    raise ValueError(f'Unknown graph feature normalization: {normalization}')
