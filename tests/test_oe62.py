import numpy as np
import pytest
from rdkit import Chem
from orbit3d.utils.oe62 import adapt_record
from orbit3d.utils.dataset_io import write_precise_sdf


def record():
    return dict(refcode_csd='synthetic',canonical_smiles='C',inchi='InChI=1S/CH4/h1H4',number_of_atoms=5,
                xyz_pbe_relaxed='5\n\nC 4 5 6\nH 4.629 5.629 6.629\nH 4.629 4.371 5.371\nH 3.371 5.629 5.371\nH 3.371 4.371 6.629\n',
                energies_occ_pbe0_vac_tier2=[-200.,-10.],energies_unocc_pbe0_vac_tier2=[1.2])


def test_source_positions_and_positive_lumo_survive_sdf(tmp_path):
    r=record();m=adapt_record(r,['C','H'])
    xyz=Chem.MolFromXYZBlock(r['xyz_pbe_relaxed']).GetConformer().GetPositions()
    np.testing.assert_array_equal(m.GetConformer().GetPositions(),xyz)
    assert float(m.GetProp('homo')) == -10.
    assert float(m.GetProp('lumo')) == 1.2  # Already eV; do not convert from Hartree.
    p=tmp_path/'source.sdf';write_precise_sdf(p,[m])
    restored=next(iter(Chem.SDMolSupplier(str(p),removeHs=False)))
    np.testing.assert_allclose(restored.GetConformer().GetPositions(),xyz,atol=1e-10)


@pytest.mark.parametrize('field,value', [('energies_unocc_pbe0_vac_tier2',[]),('number_of_atoms',4),('canonical_smiles','[CH3]')])
def test_missing_labels_and_identity_are_not_repaired(field,value):
    r=record();r[field]=value
    with pytest.raises(ValueError):adapt_record(r,['C','H'])


def test_use_source_bond_orders_in_xyz_atom_order(monkeypatch):
    from rdkit.Chem import rdDetermineBonds
    def forbidden(*a, **kw):
        raise AssertionError('Known source graph must not trigger bond-order search')
    monkeypatch.setattr(rdDetermineBonds,'DetermineBonds',forbidden)
    r=record();lines=r['xyz_pbe_relaxed'].splitlines()
    lines[2],lines[4]=lines[4],lines[2]
    r['xyz_pbe_relaxed']='\n'.join(lines)+'\n'
    m=adapt_record(r,['C','H'])
    assert [a.GetSymbol() for a in m.GetAtoms()]==['H','H','C','H','H']
    xyz=Chem.MolFromXYZBlock(r['xyz_pbe_relaxed']).GetConformer().GetPositions()
    np.testing.assert_array_equal(m.GetConformer().GetPositions(),xyz)
    assert Chem.MolToSmiles(Chem.RemoveHs(m))=='C'
