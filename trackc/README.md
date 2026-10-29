# trackc: mixed-precision frequency-domain Maxwell solver (design C1)

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

## Convention that matters
The nodal space must be H1 with phi = 0 on the PEC conductors, otherwise `K G != 0` on the free edges and the
correction fails (tested). The real bowtie export must follow the same rule when building `G`.

## Not done yet
- cuDSS / GPU backend (no GPU in the dev container; phase C-1 of the plan).
- Loading the real `.npz` exports (`A`, `b`, `G`) from the FEniCSx prototype, which are not in this repo.
- Affine frequency-independent PML assembly on the real mesh, AAA adaptive sweep, mesh morphing.
