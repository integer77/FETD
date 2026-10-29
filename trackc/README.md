# trackc: general mixed-precision frequency-domain Maxwell (H(curl)) solver (design C1)

Implements the C1 design from `trackc_frequency_domain_gpu.md`:
FP32 sparse LDL^T of the edge system + FP32 nodal gradient correction + FP64 residual refinement,
with factor recycling across nearby frequencies.

| Module | Role |
|---|---|
| `factor.py` | `Factorization(A, dtype)`: scaled complex-symmetric LDL^T. Sequential MUMPS via `mumps_shim.c` (stand-in for cuDSS, auto-compiled), SuperLU fallback. A cuDSS backend only has to match this interface. |
| `solver.py` | `GradientCorrectedPrecond` (P(r) of Section 3.3), `refine` (stationary IR), `gmres_ir` (FGMRES inside refinement, used when a factor is reused) |
| `sweep.py` | `FrequencySweep`: refactor only when k leaves `reuse_tol` of the last factor |
| `nedelec.py`, `mesh.py`, `problem.py` | Synthetic stand-in for the bowtie matrices: lowest-order Nedelec on a radially graded tet box, PEC bowtie halves + wire tip, Si half-space, impedance outer boundary + absorbing shell. Complex symmetric `A(k) = K - k^2 M - i k B` with `K G = 0`. |
| `bench.py` | `python -m trackc.bench [n_half] [power] [lambda]` |

Setup: `apt-get install libmumps-seq-dev gfortran gcc`, `pip install numpy scipy pytest`. Tests: `pytest tests`.

## General use
```python
from trackc import maxwell_system, AffineSystem, MixedPrecisionSolver
sys_ = maxwell_system(K, M, B, G, b)           # A(k) = K - k^2 M - i k B, or AffineSystem(terms, coeff_fns, G, b)
solver = MixedPrecisionSolver(sys_)            # factorization_cls=CuDSSFactorization for GPU
res = solver.solve(k)                          # b may be (n,) or (n, nrhs); nearby k reuse the FP32 factor
```
Inputs are any H(curl) discretization: frequency-independent matrices, their coefficient functions, and the
discrete gradient G. `io.py` reads and writes npz. `problem.py` is only a test fixture.
`factor_cudss.py` is an untested cuDSS backend (no GPU here).

## Convention that matters
The nodal space must be H1 with phi = 0 on the PEC conductors, otherwise `K G != 0` on the free edges and the
correction fails (tested). The real bowtie export must follow the same rule when building `G`.

## Not done yet
- Running/validating the cuDSS backend and a device-side refinement loop (no GPU in the dev container).
- Affine frequency-independent PML assembly on the real mesh, AAA adaptive sweep, mesh morphing.
