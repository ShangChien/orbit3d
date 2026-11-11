import numpy as np

from orbit3d.quantum.ocelot import OcelotTheory, frontier_orbitals, ip_objective, tuning_objective


def test_ip_tuning_uses_positive_vertical_ionization_energy():
    assert ip_objective(-100.0, -99.5, -0.5) == 0
    assert np.isclose(ip_objective(-100.0, -99.5, -0.4), 0.1)


def test_spin_frontiers_take_highest_occupied_and_lowest_unoccupied():
    energy = np.array([[-1.0, -0.4, 0.1], [-0.9, -0.2, 0.2]])
    occupancy = np.array([[1, 1, 0], [1, 0, 0]])
    assert frontier_orbitals(energy, occupancy) == (-0.4, -0.2)
    assert OcelotTheory().xc == 'HYB_GGA_XC_LC_WPBE'
    assert OcelotTheory().basis == 'def2-svp'


def test_ocelot_joint_tuning_distinguishes_neutral_lumo_and_anion_homo():
    # E(N+1)=-100.2 gives EA=+0.2 Ha; only anion HOMO satisfies -EA here.
    assert np.isclose(tuning_objective('J2', -100, -99.5, -.5, .1, -100.2, -.2), 0)
    assert np.isclose(tuning_objective('Jn2', -100, -99.5, -.5, .1, -100.2, -.2), .09)


def test_excited_scanner_checks_the_requested_one_based_root():
    from types import SimpleNamespace

    from orbit3d.quantum.excited_states import ResponseStateConvergence
    scanner = ResponseStateConvergence()
    scanner.state = 1
    scanner.base = SimpleNamespace(_scf=SimpleNamespace(converged=True), converged=np.array([True]))
    assert scanner.converged
    scanner.base.converged = np.array([False, True])
    assert not scanner.converged
    scanner.state = 2
    assert scanner.converged
    scanner.base._scf.converged = False
    assert not scanner.converged


def test_quantum_graph_failure_retains_diagnostics_and_does_not_relabel_itself():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from orbit3d.quantum.ocelot import inspect_optimized_graph

    mol = Chem.AddHs(Chem.MolFromSmiles('CC'))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    xyz = mol.GetConformer().GetPositions()
    # Separate the two methyl groups while leaving their internal geometry intact.
    right = [1] + [atom.GetIdx() for atom in mol.GetAtomWithIdx(1).GetNeighbors() if atom.GetAtomicNum() == 1]
    xyz[right] += [4., 0., 0.]
    mol.GetConformer().SetPositions(xyz)
    result = inspect_optimized_graph(mol)
    assert not result['graph_preserved'] and not result['source_graph_compatible']
    assert result['graph_check_status'] == 'incompatible_source_graph'
    assert 'bond_geometry' in result['source_graph_error']
    assert result['source_smiles'] == 'CC'


def test_quantum_graph_check_accepts_imine_stereo_addition_but_rejects_a_specified_flip():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from orbit3d.quantum.ocelot import inspect_optimized_graph

    mol = Chem.AddHs(Chem.MolFromSmiles('N=C(C)c1ccccc1'))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    result = inspect_optimized_graph(mol)
    assert result['graph_preserved']
    assert not result['changed_bonds'] and not result['changed_atom_charges']
    assert result['recovered_identity']['identity_status'] == 'stereo_specification_added'
    Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
    bond = next(b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
    bond.SetStereo(Chem.BondStereo.STEREOZ if bond.GetStereo() == Chem.BondStereo.STEREOE
                   else Chem.BondStereo.STEREOE)
    result = inspect_optimized_graph(mol)
    assert not result['graph_preserved']
    assert 'stereo_geometry' in result['source_graph_error']


def test_optimizer_trace_preserves_failed_step_units_and_source_property_meaning(tmp_path):
    import json
    from types import SimpleNamespace

    from rdkit import Chem

    from orbit3d.quantum.ocelot import (
        BOHR_ANGSTROM,
        HARTREE_EV,
        quantum_geometry_molecule,
        record_geometry_step,
    )
    record = record_geometry_step(tmp_path, 0, .2, {'coords': np.array([[0., 0., 0.], [1., 0., 0.]]),
        'gradients': np.array([[.1, 0., 0.], [-.1, 0., 0.]]), 'energy': -1.,
        'g_scanner': SimpleNamespace(converged=False)})
    assert not record['scf_converged']
    assert record['coordinates_A'][1][0] == BOHR_ANGSTROM
    assert np.isclose(record['raw_forces_eV_A'][0][0], -.1*HARTREE_EV/BOHR_ANGSTROM)
    assert json.loads((tmp_path/'geometry_cycle_00.jsonl').read_text()) == record
    mol = Chem.MolFromSmiles('CC');mol.AddConformer(Chem.Conformer(2))
    mol.SetProp('CandidateID', '7');mol.SetProp('PredictedValue', '[-7.5]')
    out = quantum_geometry_molecule(mol, record['coordinates_A'], 'geometry_not_converged')
    assert out.GetProp('CandidateID') == '7' and not out.HasProp('PredictedValue')
    assert json.loads(out.GetProp('QuantumSourceProperties'))['PredictedValue'] == '[-7.5]'
    assert out.GetProp('QuantumGraphCheck') == 'geometry_not_converged'
