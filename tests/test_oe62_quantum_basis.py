"""Check the actual OE62 worker Hamiltonian, without running an SCF calculation."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec('pyscf'),'PySCF quantum extra is not installed')
class OE62BasisTest(unittest.TestCase):
    def setUp(self):
        path=Path(__file__).parents[1]/'scripts/oe62_quantum_singlepoint.py'
        spec=importlib.util.spec_from_file_location('oe62_worker',path)
        self.worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.worker)
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)

    def test_heavy_atoms_load_matching_def2_potentials(self):
        for symbols,coords in [(['H','I'],[[0,0,0],[0,0,1.61]]),(['Te','H','H'],[[0,0,0],[1.7,0,0],[0,1.7,0]])]:
            mol=self.worker.make_mol(symbols,np.array(coords,dtype=float),Path(self.directory.name)/'heavy.log')
            self.assertEqual(mol.nelectron,26)
            heavy=next(i for i,symbol in enumerate(symbols) if symbol in ['I','Te'])
            self.assertEqual(mol.atom_nelec_core(heavy),28)
            mol.stdout.close()

    def test_light_atom_hamiltonian_matches_all_electron_def2_tzvp(self):
        from pyscf import gto
        symbols=['H','Br'];coords=np.array([[0,0,0],[0,0,1.41]])
        mol=self.worker.make_mol(symbols,coords,Path(self.directory.name)/'light.log')
        reference=gto.M(atom=list(zip(symbols,coords.tolist())),basis='def2-tzvp',spin=0,verbose=0)
        self.assertEqual(mol.nelectron,36)
        np.testing.assert_array_equal(mol.intor_symmetric('int1e_nuc'),reference.intor_symmetric('int1e_nuc'))
        mol.stdout.close()


if __name__=='__main__':unittest.main()
