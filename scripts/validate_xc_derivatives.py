"""Check native and explicit third-order CPU XC kernels on fixed spin densities."""
import argparse
import json
from pathlib import Path
import cupy
import numpy as np
from pyscf.dft.numint import NumInt as CPU
from gpu4pyscf.dft.numint import NumInt as GPU
from loom.quantum.xc_integrator import ThirdOrderCPUIntegrator
from loom.utils.run_evidence import freeze_run
p=argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a=p.parse_args()
freeze_run(a.output, __file__)
rng=np.random.default_rng(42)
rows=[]
for spin in [0,1]:
    rho=rng.normal(0,.01, (4,128) if spin==0 else (2,4,128))
    rho[0] = rng.uniform(.05,2,128) if spin==0 else rho[0]
    if spin: rho[:,0] = rng.uniform(.05,2,(2,128))
    for order in [2,3]:
        cpu,native,verified=CPU(),GPU(),ThirdOrderCPUIntegrator()
        cpu.omega=native.omega=verified.omega=.2762
        expected=cpu.eval_xc_eff('HYB_GGA_XC_LC_WPBE',rho,order)
        actual=native.eval_xc_eff('HYB_GGA_XC_LC_WPBE',cupy.asarray(rho),order)
        corrected=verified.eval_xc_eff('HYB_GGA_XC_LC_WPBE',cupy.asarray(rho),order)
        row={'spin':spin,'requested_order':order,'derivatives':[]}
        for k in range(order+1):
            x,y,z=expected[k],cupy.asnumpy(actual[k]),cupy.asnumpy(corrected[k])
            np.testing.assert_allclose(z,x,atol=1e-5,rtol=1e-4)
            row['derivatives'].append({'order':k,'native_max_abs_error':float(abs(x-y).max()),
                                      'partitioned_max_abs_error':float(abs(x-z).max())})
        rows.append(row)
r={'functional':'HYB_GGA_XC_LC_WPBE','omega':.2762,'records':rows,'passed':True,
   'execution':'Native GPU for order <=2; CPU Libxc for order >=3; no functional change'}
(a.output/'summary.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
