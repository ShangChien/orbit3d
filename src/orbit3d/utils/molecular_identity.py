"""Distinguish inferred stereochemical detail from changes to a supplied graph."""
from rdkit import Chem


def compare_molecular_identity(source, recovered):
    # RemoveHs retains hydrogens defining double-bond stereo. Removing stereo
    # afterwards therefore leaves different H graphs for an unspecified imine
    # and its specified conformer. Compare fully explicit H graphs instead;
    # this preserves both stereo-defining and isotopically labelled hydrogens.
    source = Chem.AddHs(Chem.Mol(source))
    recovered = Chem.AddHs(Chem.Mol(recovered))
    exact = Chem.MolToSmiles(source) == Chem.MolToSmiles(recovered)
    if exact:
        return {"identity_match": True, "constitution_match": True,
                "source_specified_stereo_preserved": True, "identity_status": "exact"}
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
