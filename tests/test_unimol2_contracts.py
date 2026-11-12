import numpy as np
import pytest
import torch
from rdkit import Chem
from orbit3d.oracles.unimol2_oracle import build_unimol2,source_features,collate_unimol2,grouped_batches,load_pretrained


def mol(smiles):
    m=Chem.AddHs(Chem.MolFromSmiles(smiles));c=Chem.Conformer(m.GetNumAtoms());c.SetPositions(np.random.default_rng(42).normal(size=(m.GetNumAtoms(),3)));m.AddConformer(c);return m


def test_source_heavy_coordinates_and_official_drop_features():
    m=mol('CCO');original=m.GetConformer().GetPositions().copy();f=source_features(m)
    assert f['src_token'].tolist()==[6,6,8]
    expected=original[:3].astype(np.float32);expected-=expected.mean(0)
    np.testing.assert_array_equal(f['src_pos'].numpy(),expected)
    np.testing.assert_array_equal(m.GetConformer().GetPositions(),original)
    assert torch.all(f['atom_feat']==1) and torch.all(f['shortest_path']==511)
    assert torch.all(f['pair_type'][...,0]==2) and torch.all(f['pair_type'][...,1]==130)


def test_canonical_batches_never_drop_rows_or_change_pad_length():
    lengths=np.array([2,3,4,5,6,7,8,9,10,11]);batches=grouped_batches(lengths,3,42)
    assert sorted(i for ids in batches for i in ids)==list(range(len(lengths)))
    for ids in batches:assert len({(int(lengths[i])+4)//4*4-1 for i in ids})==1
    assert batches==grouped_batches(lengths,3,42)


def test_padding_masks_and_translation_invariance():
    torch.manual_seed(7);model=build_unimol2('tiny').eval();m=mol('CCO');f=source_features(m)
    b,_=collate_unimol2([(f,np.zeros(2)),(source_features(mol('CC')),np.zeros(2))])
    assert torch.isneginf(b['attn_bias'][1,:3,3:]).all()
    assert torch.all(b['attn_bias'][1,3:,:3]==0)
    with torch.no_grad():first=model(b,classification_head_name='homo_lumo',features_only=True)[0]
    c=m.GetConformer();c.SetPositions(c.GetPositions()+np.array([2.,3.,-4.]));g=source_features(m)
    b2,_=collate_unimol2([(g,np.zeros(2)),(source_features(mol('CC')),np.zeros(2))])
    with torch.no_grad():second=model(b2,classification_head_name='homo_lumo',features_only=True)[0]
    torch.testing.assert_close(first,second,atol=1e-6,rtol=1e-6)


def test_loader_rejects_partial_backbone_before_mutation(tmp_path):
    model=build_unimol2('tiny');before=model.embed_tokens.weight.detach().clone();path=tmp_path/'partial.pt';torch.save({'model':{'embed_tokens.weight':torch.ones_like(before)}},path)
    with pytest.raises(ValueError,match='Incompatible'):load_pretrained(model,path)
    assert torch.equal(model.embed_tokens.weight,before)
