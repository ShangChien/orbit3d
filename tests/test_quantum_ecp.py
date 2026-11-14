"""Real basis/ECP contracts; PySCF is an optional quantum dependency."""
import importlib.util
import unittest
from dataclasses import asdict

import numpy as np

from loom.quantum.ocelot import OcelotTheory, ecp_core_electrons, make_quantum_molecule, theory_from_record


@unittest.skipUnless(importlib.util.find_spec('pyscf'), 'PySCF quantum extra is not installed')
class QuantumECPTest(unittest.TestCase):
    def test_heavy_def2_atoms_have_their_matching_core_potential(self):
        hi = make_quantum_molecule(['H', 'I'], [[0, 0, 0], [0, 0, 1.61]], OcelotTheory())
        self.assertEqual(ecp_core_electrons(hi), {'I': 28})
        self.assertEqual(hi.nelectron, 26)
        cation = make_quantum_molecule(['H', 'I'], [[0, 0, 0], [0, 0, 1.61]], OcelotTheory(), charge=1, spin=1)
        self.assertEqual(cation.nelectron, 25)
        teh2 = make_quantum_molecule(['Te', 'H', 'H'], [[0, 0, 0], [1.7, 0, 0], [0, 1.7, 0]], OcelotTheory())
        self.assertEqual(ecp_core_electrons(teh2), {'Te': 28})
        self.assertEqual(teh2.nelectron, 26)

    def test_light_atom_hamiltonian_is_unchanged(self):
        atoms, coords = ['H', 'Br'], [[0, 0, 0], [0, 0, 1.41]]
        new = make_quantum_molecule(atoms, coords, OcelotTheory())
        old = make_quantum_molecule(atoms, coords, OcelotTheory(ecp_policy='none'))
        self.assertEqual(new.nelectron, 36)
        self.assertEqual(ecp_core_electrons(new), {})
        np.testing.assert_array_equal(new.intor_symmetric('int1e_ovlp'), old.intor_symmetric('int1e_ovlp'))
        np.testing.assert_array_equal(new.intor_symmetric('int1e_nuc'), old.intor_symmetric('int1e_nuc'))

    def test_legacy_records_keep_their_actual_no_ecp_convention(self):
        settings = asdict(OcelotTheory())
        del settings['ecp_policy']
        legacy = theory_from_record(settings)
        self.assertEqual(legacy.ecp_policy, 'none')
        hi = make_quantum_molecule(['H', 'I'], [[0, 0, 0], [0, 0, 1.61]], legacy)
        self.assertEqual(hi.nelectron, 54)
        self.assertEqual(theory_from_record(asdict(OcelotTheory())).ecp_policy, 'matching_def2')


if __name__ == '__main__':
    unittest.main()
