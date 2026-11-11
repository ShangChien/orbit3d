"""Distinguish inferred stereochemical detail from changes to a supplied graph."""
from rdkit import Chem


def compare_molecular_identity(source, recovered):
    source = Chem.RemoveHs(Chem.Mol(source))
    recovered = Chem.RemoveHs(Chem.Mol(recovered))
    exact = Chem.MolToSmiles(source) == Chem.MolToSmiles(recovered)
    plain_source, plain_recovered = Chem.Mol(source), Chem.Mol(recovered)
    Chem.RemoveStereochemistry(plain_source)
    Chem.RemoveStereochemistry(plain_recovered)
    constitution = Chem.MolToSmiles(plain_source) == Chem.MolToSmiles(plain_recovered)
    specified_stereo = constitution and recovered.HasSubstructMatch(source, useChirality=True)
    return {
        'identity_match': exact,
        'constitution_match': constitution,
        'source_specified_stereo_preserved': specified_stereo,
        'identity_status': ('exact' if exact else 'stereo_specification_added' if specified_stereo
                            else 'specified_stereo_changed' if constitution else 'constitution_changed'),
    }
