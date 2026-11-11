"""Pass molecule-specific range separation to the supported GPU Libxc candidates."""

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
            if functional.func_id not in {478, 486}:
                raise ValueError("This adapter supports only the LC-wPBE candidates Libxc 478 and 486")
            setter(functional.xc_func, b"_omega", float(self.omega))
        return functions
