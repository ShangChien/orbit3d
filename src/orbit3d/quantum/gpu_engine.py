"""Native GPU4PySCF entry point with the verified range-separation API."""

from importlib.metadata import version
from packaging.version import Version


def gpu_scf(molecule, unrestricted=True):
    if Version(version('gpu4pyscf-cuda12x')) < Version('1.9.1'):
        raise RuntimeError('Use GPU4PySCF >=1.9.1; earlier tested builds silently dropped custom Libxc omega')
    if unrestricted:
        from gpu4pyscf.dft.uks import UKS

        return UKS(molecule)
    from gpu4pyscf.dft.rks import RKS

    return RKS(molecule)
