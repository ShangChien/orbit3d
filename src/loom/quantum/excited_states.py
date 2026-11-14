"""Spin-adapted vertical TDDFT on the same neutral geometry and tuned functional."""

import numpy as np
from .ocelot import HARTREE_EV, ecp_core_electrons, host_array


class ResponseStateConvergence:
    @property
    def converged(self):
        # TD states are numbered from 1; the root convergence array is zero-based.
        return bool(self.base._scf.converged and self.base.converged[self.state - 1])


def response_gradient_scanner(td, state=1):
    scanner = td.nuc_grad_method().as_scanner(state=state)
    if getattr(td, 'device', None) == 'gpu':
        from pyscf import lib

        # GPU4PySCF 1.9.1 indexes this array with self.state, unlike its own
        # energy/gradient access. Correct the index instead of requesting a
        # gratuitous extra root or ignoring electronic convergence.
        scanner = lib.set_class(scanner, (ResponseStateConvergence, scanner.__class__))
    return scanner


def restricted_reference(molecule, omega, theory, density=None):
    if theory.engine == 'gpu':
        import cupy
        from .gpu_engine import gpu_scf

        mf = gpu_scf(molecule, unrestricted=False)
        density = cupy.asarray(density) if density is not None else None
    else:
        from pyscf.dft.rks import RKS

        mf = RKS(molecule)
    mf.xc = theory.xc
    mf.omega = omega
    mf.grids.atom_grid = (theory.radial_grid, theory.angular_grid)
    mf.conv_tol = theory.scf_tolerance
    mf.max_cycle = theory.max_scf_cycles
    mf.kernel(dm0=density)
    if not mf.converged:
        raise RuntimeError('Restricted ground-state SCF for spin-adapted TDDFT did not converge')
    return mf


def response(mf, singlet, states=1):
    td = mf.TDDFT()
    td.singlet = singlet
    td.nstates = states
    td.conv_tol = 1e-7
    # The response solver filters eigenvalues of the trial-vector Gram matrix,
    # i.e. squared norms. Its default 1e-12 can discard corrections before a
    # 1e-7 residual has converged. Keep this threshold below the squared target.
    td.lindep = 1e-16
    td.max_cycle = 100
    energies, _ = td.kernel()
    energies = host_array(energies)
    if not np.all(td.converged) or len(energies) < states or not np.isfinite(energies).all() or (energies <= 0).any():
        raise RuntimeError(f'TDDFT response failed: singlet={singlet}, converged={td.converged}, energy_Ha={energies.tolist()}')
    return td, energies


def vertical_excitations(molecule, omega, theory, density=None, unrestricted_energy=None):
    mf = restricted_reference(molecule, omega, theory, density)
    if unrestricted_energy is not None and abs(float(mf.e_tot) - unrestricted_energy) > 1e-5:
        raise RuntimeError('Restricted and unrestricted neutral references differ; spin-adapted TDDFT is not interchangeable')
    singlet, se = response(mf, True)
    triplet, te = response(mf, False)
    return {'method': 'full linear-response TDDFT; spin-adapted restricted reference',
            'functional': theory.xc, 'basis': theory.basis, 'ecp_policy': theory.ecp_policy,
            'ecp_core_electrons': ecp_core_electrons(molecule),
            'response_residual_tolerance': singlet.conv_tol,
            'trial_gram_eigenvalue_threshold': singlet.lindep,
            'third_order_xc_engine': 'CPU Libxc (GPU SCF/response)' if theory.engine == 'gpu' else 'CPU Libxc',
            'geometry': 'same optimized S0 geometry for S1 and T1',
            'omega': omega, 'ground_energy_Ha': float(mf.e_tot),
            'singlet_eV': (se * HARTREE_EV).tolist(), 'triplet_eV': (te * HARTREE_EV).tolist(),
            'S1_minus_T1_eV': float((se[0] - te[0]) * HARTREE_EV)}
