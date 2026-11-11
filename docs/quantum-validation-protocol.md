# OCELOT quantum protocol diagnostics

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
The official 2022 supplementary PDF and 2021 OCELOT methods paper were retrieved
and checked; both name IP tuning without specifying the objective in their
methods paragraphs. The public API also implements J2 (neutral and anion HOMOs)
and Jn2 (neutral HOMO and LUMO). These are explicit diagnostic choices recorded in
the run manifest; none is silently selected based on CSV agreement.

## Reference and GPU checks (2026-11-11)

For `csd_CAKRUB11` (naphthalene), the initial Libxc486 GPU Jh tuning/optimization converged at omega
0.2723. HOMO/LUMO were -8.197418/+0.534015 eV; CSV values are
-7.387079/-0.096873 eV. The discrepancy is unresolved, so generated-molecule
cohort calculations using this protocol were stopped with partial logs preserved.

The API repository's public `tests/wtuning_test/1b` includes Gaussian16 A.03
benzene logs, allowing an independent same-geometry/6-31G/omega cross-check.
Libxc486 at omega0.2837 differs by +0.024704 Ha and -0.0160/-0.0127 eV in frontier
orbitals. Libxc478 at omega0.2219 and0.2837 gives neutral frontier differences
below0.00033 eV and energies differing by about-0.00068 Ha. The cation case still
has up to0.02684 eV orbital discrepancy. Thus the initial claim that486 exactly
maps Gaussian LC-wHPBE is withdrawn; neither name matching nor CPU/GPU parity
establishes cross-program equivalence. The reusable checker is
`scripts/validate_gaussian_reference.py`, with explicit functional and source hashes.

The initial GPU4PySCF 1.5.0 build silently omitted custom omega when initializing
Libxc. That diagnostic first exposed a 5.30 Ha CPU/GPU mismatch. The temporary
local Libxc patch has now been removed: the supported GPU extra pins official
GPU4PySCF CUDA12x 1.9.1 / GPU Libxc 0.9.0, which forward omega natively.
At the same geometry and omega, the unpatched official engine agrees with CPU
PySCF to 8.996e-10 Ha and 1.34e-6 eV in frontier orbitals (no density fitting).
See `experiments/quantum_gpu19_parity_20261111.json`.

The previous full-TDDFT failure was a solver-threshold conflict. A residual target
of 1e-7 with the default trial Gram eigenvalue threshold 1e-12 can discard
corrections before convergence. With the latter set to 1e-16, full spin-adapted
singlet and triplet TDDFT converge at unchanged functional, basis, omega and grid.
For the J2/478 naphthalene geometry (omega0.2674), vertical S1=4.6019275 eV,
T1=2.6072350 eV and S1−T1=1.9946925 eV. These are vertical diagnostic results,
not adiabatic gaps or validation against the OCELOT CSV. The restricted reference
must agree with the neutral unrestricted ground energy within1e-5 Ha.
Ground-state results and optimized geometry are written before optional excited
states so an excited-state failure does not erase an already completed ground run.

## Exact implementation choices

- The initial diagnostic used Libxc `HYB_GGA_XC_LC_WPBE_WHS` (486). The alternative
  `HYB_GGA_XC_LC_WPBE` (478) is now the explicit default based on the two neutral
  public Gaussian checks above; the cation discrepancy remains unresolved.
  Both have zero SR and 100% LR HF. Their identity with the reference implementation
  is not assumed.
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

On Linux/CUDA12, `uv sync --extra quantum-gpu` selects the tested native GPU
versions. Use `--engine gpu --workers 1 --threads 2 --excitations` to evaluate
all supplied molecules at this explicit diagnostic level. CPU and GPU response
solvers share the same convergence thresholds. The optional
`scripts/validate_excited_states.py` accepts a saved `--ground-run`, preserving
its geometry, functional, basis and omega instead of retuning for each state.

J2 and Jn2 diagnostics with Libxc478 also failed to reproduce naphthalene's CSV
frontier orbitals: J2 gave −8.15746/+0.53337 eV and Jn2 −8.15902/+0.53452 eV.
Changing the tested tuning objective does not resolve the dataset discrepancy.
