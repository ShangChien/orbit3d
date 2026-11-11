"""Reconstructed input geometry for the OCELOT CSV workflow."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from ..generator.octree_tokenizer import rdkit_mol_to_mol3d_instance, mol3d_to_validated_rdkit_mol


def ocelot_conformer(smiles, seed=42):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError('Unparseable source SMILES')
    identity = Chem.MolToSmiles(molecule, isomericSmiles=True)
    molecule = Chem.AddHs(molecule)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    parameters.maxIterations = 300
    if AllChem.EmbedMolecule(molecule, parameters):
        raise ValueError('ETKDG embedding failed')
    try:
        if not AllChem.MMFFHasAllMoleculeParams(molecule):
            raise ValueError('MMFF parameters unavailable')
        status = AllChem.MMFFOptimizeMolecule(molecule, maxIters=100)
        if status < 0:
            raise ValueError('MMFF setup failed')
        checked = mol3d_to_validated_rdkit_mol(rdkit_mol_to_mol3d_instance(molecule, np.empty(0)))
    except (Chem.KekulizeException, Chem.AtomValenceException) as exc:
        raise ValueError('MMFF aromaticity/valence unsupported: ' + str(exc)) from exc
    if Chem.MolToSmiles(Chem.RemoveHs(checked), isomericSmiles=True) != identity:
        raise ValueError('Conformer preparation changed the source chemical identity')
    molecule.SetIntProp('MMFFOptimizationStatus', int(status))
    molecule.SetProp('GeometryOrigin', 'ETKDGv3 + MMFF94 (100-step limit); reconstructed, not original DFT')
    return molecule
