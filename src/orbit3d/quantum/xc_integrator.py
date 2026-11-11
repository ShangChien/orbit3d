"""GPU quadrature with explicit CPU Libxc evaluation of third-order derivatives.

GPU Libxc 0.9's LC-wPBE third-order call loses the exchange contribution, including
its lower derivatives. SCF/linear response use the verified native first/second
derivatives; excited-state gradients need the complete CPU third-order kernel.
This changes execution location, not the functional, basis, grid or omega.
"""

import cupy
from gpu4pyscf.dft.numint import NumInt


class ThirdOrderCPUIntegrator(NumInt):
    def eval_xc_eff(self, xc_code, rho, deriv=1, *, omega=None, xctype=None,
                    verbose=None, spin=None, **kwargs):
        if deriv < 3:
            return super().eval_xc_eff(xc_code, rho, deriv, omega=omega, xctype=xctype,
                                       spin=spin, **kwargs)
        result = self.to_cpu().eval_xc_eff(
            xc_code, cupy.asnumpy(rho), deriv, omega=self.omega if omega is None else omega,
            xctype=xctype, verbose=verbose, spin=spin,
        )
        return [cupy.asarray(value) if value is not None else None for value in result]
