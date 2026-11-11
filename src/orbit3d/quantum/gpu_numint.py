"""Pass molecule-specific range separation to GPU Libxc 486."""

import ctypes
from functools import lru_cache
from gpu4pyscf.dft import numint, libxc


class OcelotNumInt(numint.NumInt):
    @lru_cache(10)
    def _init_xcfuns(self, xc_code, spin=0):
        functions = numint._init_xcfuns(xc_code, spin)
        setter = libxc._libxc.xc_func_set_ext_params_name
        setter.argtypes = (libxc._xc_func_p, ctypes.c_char_p, ctypes.c_double)
        setter.restype = None
        for functional, weight in functions:
            if functional.func_id != 486:
                raise ValueError("This adapter is specifically for IP-tuned LC-wHPBE (Libxc 486)")
            setter(functional.xc_func, b"_omega", float(self.omega))
        return functions
