import pytest
from rdkit import Chem
from loom.baselines.brics import build_library,FragmentSampler


def test_single_cut_recombination_has_no_dummies_and_replays_by_candidate_seed():
    molecules=[]
    for i,smiles in enumerate(['CCCOCc1ccccc1','CCNC(=O)c1ccncc1','CCOc1ccc(NC)cc1']):
        mol=Chem.MolFromSmiles(smiles);mol.SetProp('SourceID',str(i));molecules.append(mol)
    library=build_library(molecules);sampler=FragmentSampler(library)
    first=[sampler.sample(i) for i in range(12)]
    assert first==[sampler.sample(i) for i in range(12)]
    assert first[::-1]==[sampler.sample(i) for i in reversed(range(12))]
    for smiles,record in first:
        mol=Chem.MolFromSmiles(smiles)
        assert len(Chem.GetMolFrags(mol))==1
        assert all(a.GetAtomicNum()>0 for a in mol.GetAtoms())
        left,right=[Chem.MolFromSmiles(s) for s in record['fragment_smiles']]
        assert mol.GetNumAtoms()==left.GetNumAtoms()+right.GetNumAtoms()-2
    assert all(set(r['source_ids']) <= {'0','1','2'} for r in library['fragments'])


def test_brics_baseline_does_not_replace_an_empty_training_library():
    mol=Chem.MolFromSmiles('c1ccccc1');mol.SetProp('SourceID','benzene')
    with pytest.raises(ValueError,match='No single-cut'):
        build_library([mol])
