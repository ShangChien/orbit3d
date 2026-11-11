# OCELOT reference-level quantum validation

The OCELOT reference is molecule-specific IP-tuned LC-ωHPBE/def2-SVP, with
geometry optimization at that level, rather than a fixed-ω generic PBE hybrid.
[OCELOT paper](https://pubs.rsc.org/en/content/articlehtml/2023/sc/d2sc04676h),
[official methodology](https://oscar-docs.as.uky.edu/IP%20and%20EA/).

The public [OCELOT tuning implementation](https://github.com/caer200/ocelot_api/blob/c0bc4083e5deb61ff1eb71d119a7a5dcfb2abab0/ocelot/task/wtuning.py)
defaults to the Jh objective `abs(epsilon_HOMO + E_cation - E_neutral)` and alternates
range-parameter tuning with geometry optimization. We implement this in PySCF
2.12.1 with geomeTRIC 1.1. The default is not evidence of the dataset production
configuration. The naphthalene reference cross-check below fails: this remains a
diagnostic at the nominal functional/basis, not validated reproduction of OCELOT.
Original Gaussian output, per-molecule omega and original geometries are absent
from the available CSV. Do not substitute a tuning objective merely to fit labels.

## Reference and GPU checks (2026-11-11)

For `csd_CAKRUB11` (naphthalene), GPU Jh tuning/optimization converged at omega
0.2723. HOMO/LUMO were -8.197418/+0.534015 eV; CSV values are
-7.387079/-0.096873 eV. The discrepancy is unresolved, so generated-molecule
cohort calculations using this protocol were stopped with partial logs preserved.

GPU4PySCF 1.5.0's numerical integrator does not forward custom omega to its
cached Libxc objects. `quantum/gpu_numint.py` explicitly sets Libxc 486's `_omega`
parameter. Without it, a fixed-geometry CPU/GPU comparison differed by 5.30 Ha.
With it, energy differs by -1.08e-10 Ha and HOMO/LUMO by 8.28e-6/1.21e-5 eV.
This checks engine parity, not dataset protocol fidelity. The measured stack is
PySCF 2.12.1, GPU4PySCF CUDA12x 1.5.0, GPU Libxc 0.5, CuPy 13.4.1,
cuTENSOR 2.2.0 and NumPy 2.2.6; no density fitting. GPU jobs use `--engine gpu`.

## Exact implementation choices

- Libxc `HYB_GGA_XC_LC_WPBE_WHS` (486), Henderson/Janesko/Scuseria short-range
  exchange and long-range HF, with PBE correlation. Zero SR and 100% LR HF.
- Gaussian's LC-w**H**PBE naming must not be confused with Libxc LC-wPBE**h**
  (487), which has 25% short-range HF. The original Vydrov LC-wPBE (478) is also
  a distinct exchange-hole implementation. [Libxc functional entries in CP2K](https://manual.cp2k.org/cp2k-2026_2-branch/CP2K_INPUT/FORCE_EVAL/PROPERTIES/TDDFPT/XC/XC_FUNCTIONAL.html).
- def2-SVP, gas phase, explicit hydrogen coordinates, UKS neutral singlet and
  cation doublet; no density-fitting approximation or dispersion addition.
- Initial ω bounds [0.05, 0.5] bohr^-1, bounded minimization tolerance 0.001,
  four-decimal ω precision. Later geometry/tuning cycles search ±0.2 around
  the current ω, restricted to positive values. Outer-loop tolerance 0.01.
- SCF energy tolerance 1e-9 Hartree, at most100 cycles; integration grid
  99 radial/590 angular points with PySCF's pruning. Geometry gradient RMS
  threshold3e-4 Hartree/bohr and maximum4.5e-4; max80 steps.
- Every molecule records tuning points, charge-state energies, spin-square,
  final orbitals, raw DFT forces, geometry and graph identity. Nonconverged
  calculations remain failures. No cheaper functional is substituted.

These numerical integration/optimizer settings are recorded explicitly; they
are not asserted to reproduce every Gaussian16 default. Optimized and fixed-input
geometry calculations are separate endpoints. HOMO/LUMO orbital eigenvalues are
not ΔSCF IP/EA or excitation energies. TDDFT excitation energies and adiabatic
singlet-triplet differences require their own calculations and state labels.

## Run

```bash
uv sync --extra quantum
uv run --extra quantum python scripts/validate_quantum.py \
  --sdf outputs/run/generated.sdf --output outputs/quantum \
  --workers 8 --threads 4
```

Use `--fixed-geometry` only for a separately labelled single-geometry evaluation.
The initial validation cohort is fixed to all16 retained molecules from
`octree_mainline_v1`, plus the naphthalene entry `csd_CAKRUB11` for a reference
cross-check. Model scores do not select a favorable subset for quantum reporting.
