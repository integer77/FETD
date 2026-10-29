# Open-source GPU full-wave solvers for multi-scale near-field electromagnetics

State of the art, gap analysis, technical options and a development plan

*Written October 2026. The reference problem is the THz bowtie antenna with a near-field probe tip from the bowtie-snom project. Measured numbers come from the CST runs and the FEniCSx prototype. Everything marked as an estimate is an order-of-magnitude calculation that a prototype must confirm.*

---

## 0. Summary

- **The real obstacle is scale separation, not GPUs.** The bowtie problem spans about 4.5 orders of magnitude, from 10 nm mesh features to a 333 µm wavelength. Every explicit time-domain method (FDTD, FIT, explicit DGTD) pays for that ratio in time steps. Every frequency-domain method pays for it in an ill-conditioned, indefinite, complex linear system.
- **Explicit GPU solvers are abundant in open source, but they solve a different problem.** At least a dozen GPU FDTD codes exist (FDTDX, Khronos.jl, gprMax, fdtdz, Luminescent.jl and others). They target optical photonics, where the scale separation is 10² to 10³. On Cartesian grids they staircase curved metal surfaces, and their cost grows linearly with scale separation.
- **The best existing open-source starting point is Palace** (AWS, built on MFEM). It already has curved high-order tetrahedra, an implicit time-domain solver with AMS-preconditioned CG, p-multigrid, and NVIDIA or AMD GPU support. It lacks a PML, a plane-wave or scattered-field excitation, time-domain surface impedance, and a single-precision path for consumer GPUs. For open-region scattering problems such as s-SNOM, those gaps are decisive.
- **Recommended architecture for high scale separation.** Use an implicit FETD (time-domain finite element) solver on curved tetrahedra. Each time step solves a real, symmetric positive definite system with conjugate gradients and an AMS preconditioner, warm-started from the previous step. This system is exactly the kind AMS handles robustly, it is memory-bound (so FP64 is affordable even on consumer GPUs), and it needs no factorization (so it fits in 8 GB of VRAM). A broadband run with on-the-fly DFT replaces a 20 to 30 point frequency sweep.
- **Estimated payoff on your laptop.** The estimate is about 1 to 2 min per scan point on the RTX 2000 Ada, against 7 to 10 min for the CPU frequency-domain sweep and about 18 min for CST time domain. On a data-centre GPU the same code would be another 10 to 30 times faster, because the method is limited by memory bandwidth.
- **Highest community value at the lowest risk.** First prototype the method in FEniCSx with a real-valued PETSc and hypre build. Then contribute a time-domain PML, scattered-field plane-wave excitation and time-domain metal models to Palace. A new code from scratch is only justified for the explicit DGTD track, which suits low-scale-separation photonics.
- **Genuinely open research problems.** These are robust GPU preconditioners for indefinite, PML-loaded frequency-domain Maxwell problems, FP32-first solvers with rigorous error control for consumer GPUs, and scan workflows that keep the discretization error smooth as a probe moves.

---

## 1. The reference problem and why it is hard

### 1.1 Physical setup

| Item | Value |
|---|---|
| Antenna | Au bowtie, 70 µm equilateral arms, 200 nm thick, gap 30 to 100 nm, tip rounding 20 nm |
| Substrate | HR-Si, ε ≈ 11.68, 1 to 50 µm thick |
| Probe | PEC cone, 15° half-angle, 200 µm long, apex radius 20 to 50 nm, 40 to 100 nm above the Au |
| Excitation | Single-cycle THz pulse, 0.2 to 2 THz, plane wave at 45° from the back, 10° tilt |
| Output | Local field enhancement under the apex as a function of tip position, 70 × 140 scan grid (9,800 points) |

### 1.2 Scale separation

Define the scale separation as $S = \lambda_{\min} / h_{\min}$, the shortest wavelength of interest divided by the smallest element size that the geometry forces.

- At 2 THz, $\lambda_0 = 150$ µm. In Si, $\lambda = 44$ µm.
- The apex and gap need $h_{\min} \approx 5$ to 15 nm.
- So $S \approx 10^4$ (in Si) to $3 \times 10^4$ (in air).

For comparison, typical values in other fields are listed below.

| Application | λ | $h_{\min}$ | $S$ |
|---|---|---|---|
| Silicon photonics, waveguides and gratings | 1.55 µm (0.45 µm in Si) | 10 to 20 nm | 20 to 100 |
| Plasmonic nanoantennas in the visible | 0.8 µm | 1 to 5 nm | 10² to 10³ |
| Metasurfaces | 0.5 to 10 µm | 10 to 50 nm | 10 to 10³ |
| **THz near-field (this problem)** | 150 to 1,500 µm | 5 to 15 nm | **10⁴ to 10⁵** |
| Superconducting qubits and on-chip RF | 1 to 10 cm | 0.1 to 1 µm | 10⁴ to 10⁶ |

Two observations follow.

1. **The THz near-field problem sits in the same regime as on-chip RF and microwave design**, not in the optical photonics regime. RF design is dominated by implicit and frequency-domain FEM (HFSS, Palace), not by FDTD, and for good reason.
2. **Open-source GPU development has concentrated on the photonics regime**, where explicit FDTD is efficient. This mismatch is the core of the gap.

### 1.3 Measured costs so far

| Run | Discretization | Cost | Source |
|---|---|---|---|
| CST time domain, 30 nm gaps | 3.5M hex cells, 5 nm local cells | dt about 0.01 fs, impractical for scans | Earlier chat |
| CST time domain, 100 nm gaps | 1.6M hex cells, 20 nm local cells | 84,009 steps at 11.5 ms, about 18 min per point on 8 threads | CST solver log |
| FEniCSx frequency domain (prototype) | 236k curved tets, 289k N1curl DOFs | 45 s and 6 GB per frequency on 2 cloud cores | Prototype test |
| Same, extrapolated to laptop | 6 MPI ranks on i7-13800H | about 15 to 25 s per frequency, 7 to 10 min for 20 to 30 frequencies | Estimate |

The prototype also showed the key quality result. With curved tets and a tip-anchored size field, the averaged readouts converge smoothly. $|E_\text{ball}|$ went from 248 to 234 as the element size went from 30 nm to 7 nm, and a 5 to 10 nm tip shift changed it smoothly (238 to 244). Point values still scatter by about ±10 %.

### 1.4 Requirements for a solver

1. Mesh-independent local enhancement at a nm-scale apex, which needs conformal curved elements and averaged readouts.
2. Broadband response, 0.2 to 2 THz, from one simulation or a cheap sweep.
3. Oblique plane-wave incidence on a layered substrate.
4. Open boundaries (PML).
5. Metals, with a PEC tip and Au as PEC, a surface impedance or a Drude volume model.
6. About 10³ to 10⁴ scan positions, so a few minutes per point at most.
7. Runs on a laptop GPU (FP64 at 1:64, 8 GB VRAM) and scales to data-centre GPUs and clusters.

---

## 2. Method families and how they map to GPUs

### 2.1 Overview

| Method | Mesh | Per-step or per-frequency work | Dominant kernel | Precision need | Cost growth with $S$ | GPU fit |
|---|---|---|---|---|---|---|
| FDTD / FIT (Yee) | Cartesian, graded | Explicit stencil update | Stencil, memory-bound | FP32 fine | Linear (global CFL), plus cell count | Excellent |
| Conformal FDTD, subgridding | Cartesian with cut cells or nested grids | Explicit | Stencil | FP32, stability care | Reduced, but subgridding is fragile | Good |
| Explicit DGTD | Unstructured, curved tets or hexes | Explicit, element-local | Small dense matrix products, compute-bound | FP32 fine | Linear in fine elements, softened by LTS | Excellent |
| Implicit FETD | Unstructured, curved tets | One SPD sparse solve per step | SpMV plus AMG, memory-bound | FP64 outer, FP32 inner possible | Only through conditioning, which AMS absorbs | Good |
| Frequency-domain FEM, direct | Unstructured | One factorization per frequency | Dense supernodal BLAS3, compute-bound | FP64, or FP32 plus refinement | Memory grows superlinearly | Poor on consumer GPUs |
| Frequency-domain FEM, iterative | Unstructured | Krylov plus preconditioner per frequency | SpMV plus preconditioner | FP64 | Preconditioner degrades with frequency and PML | Open research |
| BEM / MoM with FMM or H-matrices | Surfaces only | Compressed dense solve | Compute-bound | FP64 | Moderate | Good on data-centre GPUs |
| Quasi-static near zone coupled to full wave | Two-level | One full-wave solve plus fast local solves | Mixed | FP64 | Very low | Good |

### 2.2 Why explicit methods break down at high $S$

For an explicit scheme the stable time step is set by the smallest element.

$$\Delta t_\text{FDTD} \approx \frac{h_{\min}}{c\sqrt{3}}, \qquad \Delta t_\text{DG} \approx C\,\frac{r_\text{in,min}}{c\,(p+1)^2}$$

Here $r_\text{in} \approx 0.2\,h$ is the inscribed radius of a tetrahedron, $p$ is the polynomial order and $C \approx 0.5$ for low-storage Runge-Kutta. The number of steps for a simulated time $T$ is $N_t = T/\Delta t$, which grows linearly with $S$ for a fixed number of periods. Local time stepping (LTS) limits the cost of small steps to the small elements, but the fine elements still take $N_t \propto S$ steps.

### 2.3 Why implicit time domain does not

An implicit scheme (Newmark with β = 1/4, generalized-α, Crank-Nicolson) is unconditionally stable. The time step is set by the bandwidth of the pulse, typically 20 to 40 steps per shortest period, and does not depend on the mesh. The cost moves into the linear system

$$\left(M_\varepsilon + \gamma\,\Delta t\, C_\sigma + \beta\,\Delta t^2 K\right) e^{n+1} = \text{rhs}^n ,$$

with $M_\varepsilon$ the permittivity-weighted mass matrix, $C_\sigma$ the conductivity (and PML damping) matrix, and $K$ the curl-curl stiffness matrix.

- **It is real, symmetric and positive definite**, unlike the complex, indefinite frequency-domain matrix $K - \omega^2 M$.
- **Small elements make $K$ dominate** (ratio $(c\Delta t/h)^2 \sim 10^5$ to $10^6$ near the tip). This is the near-singular curl-curl regime that the Hiptmair-Xu auxiliary space preconditioner (AMS) was designed for, and it is robust there.
- **The matrix is constant** in time, so the preconditioner setup is done once per scan point.
- **Warm start.** The solution from the previous step (or an extrapolation from the last two) is an excellent initial guess, so few CG iterations are needed per step.

This is the key insight of this document. **Solving the time-domain problem implicitly turns the hard indefinite frequency-domain linear algebra into easy positive definite linear algebra, and a DFT of the time trace recovers all frequencies.** This is how commercial RF codes avoid the problem in their transient solvers, and Palace already uses it (AMS with CG for transient runs by default).

### 2.4 Hardware realities

| Quantity | Laptop RTX 2000 Ada | Data-centre GPU (H100 class) | Laptop CPU (i7-13800H, DDR5-5200) |
|---|---|---|---|
| Memory bandwidth | about 256 GB/s | 2 to 3.4 TB/s | about 83 GB/s |
| FP32 compute | about 10 TFLOP/s class | about 60 TFLOP/s | about 1 TFLOP/s |
| FP64 compute | 1/64 of FP32 (a few hundred GFLOP/s) | about 1/2 of FP32 | about 1/2 of FP32 |
| Memory | 8 GB | 80 to 192 GB | 32 GB |

The design consequences are these.

- **Memory-bound kernels are insensitive to the FP64 ratio.** SpMV, stencils, low-order matrix-free operators and vector updates move about 12 bytes per 2 flops. Even a 1:64 FP64 unit keeps up with 256 GB/s. So **an FP64 sparse iterative solver is viable on consumer GPUs**, and its speed-up over a CPU equals the bandwidth ratio (about 3× on this laptop, 25 to 40× on data-centre GPUs).
- **Compute-bound kernels need FP32 on consumer GPUs.** Dense factorization, high-order DG volume kernels and MoM fall in this group. Here FP32 with error control (iterative refinement, compensated summation) is mandatory.
- **The FP64 trend is down, not up.** NVIDIA's Blackwell Ultra (B300) reportedly cuts native FP64 to about 1.2 TFLOP/s, a 1:64 ratio like consumer parts. Research on FP64 emulation through FP8 or INT8 tensor cores (the Ozaki scheme, supported in cuBLAS since 2025) projects large gains for dense kernels, but these are model projections and do not help memory-bound sparse kernels. A solver designed today should not rely on fast native FP64.

---

## 3. State of the art

### 3.1 Commercial tools

| Tool | Main EM methods | GPU support (current documentation) | Notes for this problem |
|---|---|---|---|
| CST Studio Suite 2026 | FIT and TLM time domain (hex), FEM frequency domain (tet), MoM/MLFMM | NVIDIA GPUs for time domain (FIT, TLM), integral equation (MoM, MLFMM), frequency domain **DDM only**, asymptotic, PIC. AMD GPUs only for time domain FIT. Frequency-domain DDM, IE and asymptotic solvers require GPUs with good FP64 performance (B200 or H100 named). | The time-domain solver staircases curved metal. The frequency-domain tet solver is the right method, but its GPU path needs data-centre cards. |
| Ansys HFSS 2025 R1 | FEM frequency domain, FEM transient, SBR+ | Frequency-domain and time-domain solvers and SBR+ on NVIDIA CUDA workstation and data-centre cards | The industry reference for high-$S$ RF problems. Curved tets, DDM. |
| COMSOL 6.3 / 6.4 | FEM frequency and time domain, DG time-explicit for some physics | 6.4 adds NVIDIA cuDSS as a GPU direct solver (single or double precision, multi-GPU). 6.3 added GPU time-explicit pressure acoustics. | cuDSS can serve any FEM model that uses a direct solver, limited by VRAM. |
| Ansys Lumerical FDTD | FDTD | CUDA GPU solver | Optical photonics focus. |
| Flexcompute Tidy3D | FDTD (GPU, cloud) | Cloud only, commercial engine, open-source Python client | Very fast, but cannot run locally. Cartesian grid with subpixel averaging. |

### 3.2 Open-source FDTD and related codes

| Code | Language | GPU | License | Notes |
|---|---|---|---|---|
| Meep | C++, Python | No (MPI) | GPL-2.0+ | Most mature, adjoint gradients, uniform grid only |
| openEMS | C++, MATLAB, Python | No (OpenMP, MPI) | GPL-3.0 | EC-FDTD, graded grids, RF focus |
| gprMax | Python, Cython | CUDA | GPL-3.0 | Ground-penetrating radar, reported 30× CUDA speed-up |
| FDTDX | Python, JAX | Multi-GPU | MIT | Differentiable 3D FDTD for inverse design, very active |
| Khronos.jl | Julia | CUDA, ROCm, Metal, oneAPI | MIT | Vendor-independent, differentiable |
| fdtdz | C++, CUDA, JAX | CUDA | MIT | Very fast, but limited to 2.5D and simple dielectrics |
| Luminescent.jl | Julia | CUDA | MIT | Differentiable, early stage |
| fdtd3d | C++ | CUDA, MPI | GPL-3.0 | Cross-architecture |
| flaport/fdtd | Python, PyTorch | CUDA | MIT | Educational |
| GSvit | C, C++ | CUDA | GPL-2.0 | Nanoscale optics including SNOM |

Common limitations for this problem are Cartesian grids (staircased or subpixel-averaged metal), no or limited local refinement, and a global time step. None of these codes can resolve a 20 to 50 nm apex inside a 300 µm domain at acceptable cost. The survey cited in the references gives a fuller comparison.

### 3.3 Open-source finite element codes

**Palace (AWS Labs, Apache-2.0, built on MFEM)**

- *Problem types.* Frequency-domain driven (lumped ports, numeric wave ports, surface currents), eigenmode with losses, implicit transient, electrostatic and magnetostatic extraction, 2D mode analysis.
- *Discretization.* Arbitrary order Nédélec elements, curved meshes, adaptive mesh refinement (not for transient).
- *Transient integrators.* Generalized-α (default, ρ∞ = 1), two-stage SDIRK, SUNDIALS ARKODE and CVODE with adaptive steps. No explicit schemes.
- *Linear solvers.* AMS (default for time domain, with CG), BoomerAMG, real sparse direct (SuperLU, STRUMPACK, MUMPS) used on a real approximation of the complex matrix for frequency-domain problems, geometric p- and h-multigrid with Chebyshev smoothing, shifted positive definite preconditioner options for frequency domain.
- *GPU.* CUDA or HIP via MFEM and libCEED, partial assembly on tets, prisms, pyramids and hexes, multi-GPU.
- *Frequency sweeps.* Adaptive fast sweep (projection-based model order reduction).
- *Boundary conditions.* PEC, PMC, impedance (RLC), rational impedance (frequency domain), finite conductivity (frequency domain and eigenmode only), first-order absorbing (all solvers), second-order absorbing (frequency domain only), periodic and Floquet, lumped, wave and Floquet ports, surface current.
- *Missing for scattering.* PML is listed as "coming soon" and is not implemented for either formulation. There is no plane-wave or scattered-field excitation except the (0,0) order of a Floquet port. There is no time-domain surface impedance, and no single-precision mode is documented.

Palace was built for superconducting quantum hardware, which sits in the same high-$S$ regime as THz near-field problems. Its transient solver is already the architecture recommended in Section 5. **The missing pieces are exactly what open-region scattering needs.**

**MFEM.** This is the library under Palace. It provides Nédélec elements of any order, curved meshes, GPU partial assembly through libCEED, hypre AMS and ADS, and many Maxwell examples. Building directly on MFEM gives full control at the cost of writing the driver and I/O.

**FEniCSx (DOLFINx).**
- *Strengths.* Python-level productivity, gmsh integration and complex arithmetic. It provides the discrete gradient that AMS needs (`dolfinx.fem.discrete_gradient`, checked in the 0.11 build). This is what the bowtie prototype uses.
- *GPU.* It runs through PETSc's CUDA or HIP backends. The cuDOLFINx extension moves assembly to the GPU (as of its last release it required dolfinx 0.10 and a PETSc built from source with CUDA).
- *Constraint.* hypre (and so AMS) does not work with complex PETSc scalars. **A time-domain formulation is real-valued, so this restriction disappears.** That makes FEniCSx a convenient prototyping platform for the implicit FETD route.

**NGSolve.** Strong high-order and hybrid DG (HDG) Maxwell capabilities on CPU. Its GPU support is limited, so treat it as a reference for high-order formulations and PML implementations rather than as a GPU platform.

**deal.II.** Provides excellent matrix-free GPU infrastructure, especially on hexahedra, but has no ready-made Maxwell scattering solver.

### 3.4 Discontinuous Galerkin time domain (DGTD)

- **Foundations.** Nodal DG for Maxwell (Hesthaven and Warburton) maps naturally to GPUs (Klöckner, Warburton, Bridge and Hesthaven 2009). Each element update is a set of small dense matrix products with high arithmetic intensity.
- **Multirate and local time stepping on GPUs.** Gödel, Schomann, Warburton and Clemens (2010) showed multirate Adams-Bashforth DG on GPUs. More recently Lizarazo and Silva (2024) reported a parallel-GPU DGTD with a third-order local time stepping scheme that cut run time by about 78 % on a multi-scale scattering problem at unchanged accuracy.
- **Nanophotonics.** DG has been developed extensively for nanophotonics (Busch, König and Niegemann 2011 review). Inria's DIOGENeS suite is a CPU DGTD package for nanophotonics.
- **Lessons from seismology.** SeisSol and EDGE solve elastic waves with ADER-DG on unstructured tets with clustered rate-2 local time stepping. Breuer and Heinecke report LTS speed-ups of 2.1× over global stepping on a 743k-element mesh, and they note that SeisSol's GPU port is efficient with global time stepping but only mediocre with LTS. **LTS on GPUs is still an engineering challenge**, because clusters of very different size give poor load balance.
- **Gap.** I could not find a maintained open-source GPU DGTD package for Maxwell with curved tets, LTS, PML and dispersive materials. The published GPU DGTD codes are research codes.

### 3.5 Linear algebra

**Sparse direct solvers**
- *MUMPS.* CPU only, block low-rank compression (BLR). It is the workhorse of the prototype.
- *NVIDIA cuDSS.* Real and complex, single and double precision. It can downcast double-precision factors to single precision and run iterative refinement (`CUDSS_CONFIG_DOWNCAST_FACTORS`, `CUDSS_CONFIG_IR_N_STEPS`). A hybrid memory mode keeps part of the factor on the host. In a recent thermal-simulation benchmark (CUTh-Solver, 2026) it ran out of memory before the iterative methods and was slower than a CPU iterative solver for a single right-hand side, because its setup dominated.
- *STRUMPACK and SuperLU_DIST.* Both have GPU support, and STRUMPACK adds rank-structured compression. They are relevant for frequency-domain subdomain solves.

**Algebraic multigrid**
- *hypre BoomerAMG.* Runs on NVIDIA GPUs, with HIP and SYCL ports.
- *AMS (Hiptmair and Xu 2007, Kolev and Vassilevski 2009).* Built on BoomerAMG and optimal for $\alpha\,\nabla\times\nabla\times + \beta I$ with $\alpha, \beta > 0$, which is exactly the implicit FETD matrix. Recent hypre versions run it on the device, but check the version you build.

**Preconditioners for indefinite, frequency-domain Maxwell (the open problem)**
- *Shifted or damped operators* as preconditioners (complex shifted Laplace type). Palace's `PCMatShifted` and `PCMatReal` options are of this kind. Iteration counts grow with frequency and domain size.
- *Optimized Schwarz domain decomposition* with impedance transmission conditions (Dolean, Gander and Gerardo-Giorda 2009) and *GenEO-type coarse spaces* for indefinite problems (Bootland, Dolean, Graham, Ma, Scheichl and Spence). These are promising, and the subdomain solves map to batched GPU direct solvers.
- *Sweeping preconditioners* (Tsuji, Engquist and Ying 2012 for Maxwell), plus FETI-DPEM and the domain decomposition used in HFSS. These are effective but complex to implement.
- *EM-WaveHoltz* (Peng and Appelö, IEEE TAP 2022). It computes time-harmonic solutions by filtering time-domain simulations, which yields a positive definite system solvable by CG or GMRES, with convergence guaranteed away from resonances. **It lets a GPU time-domain kernel solve frequency-domain problems**, a natural fit for this architecture.
- *Mixed precision.* GMRES-based iterative refinement in three precisions (Carson and Higham 2018) is the theory behind FP32 factorization with FP64-accurate results.

### 3.6 Frequency sweeps and broadband response

- **Model order reduction.** Palace's adaptive fast sweep and the fast sweeps in HFSS and CST reduce a sweep to a few full solves plus a reduced model. AAA rational approximation (Nakatsukasa, Sète and Trefethen 2018) is a simple, robust alternative for scalar outputs such as the gap field.
- **Time domain plus DFT.** A single broadband pulse with on-the-fly DFT at the probe points gives the whole spectrum in one run. With an implicit scheme the number of steps is set by bandwidth and duration only.

### 3.7 Plane waves on layered substrates

- *FDTD.* Total-field/scattered-field (TF/SF) boxes, with a 1D auxiliary grid for layered backgrounds.
- *FEM, frequency domain.* Scattered-field formulation with the analytic layered background, as in `slab_background.py` of the prototype.
- *FEM, time domain.* Same formulation with the background obtained by inverse FFT of the frequency-domain layered solution times the pulse spectrum. Alternatively, a Huygens surface (TF/SF on an FEM surface).
- *Periodic structures.* Floquet ports (Palace has these for frequency domain).

---

## 4. Gap analysis

| # | Gap | Evidence | Who is affected |
|---|---|---|---|
| G1 | **No open-source GPU solver for open-region scattering with curved tets, PML and plane-wave excitation.** | Palace has GPU and curved tets but no PML or plane wave. FEniCSx has the flexibility but an immature GPU path. GPU FDTD codes use Cartesian grids. | Near-field microscopy, antenna RCS, nanoantennas, metamaterial unit cells in open space |
| G2 | **High scale separation is unserved on GPUs in open source.** | Open GPU development targets photonics ($S \le 10^3$). Explicit cost grows linearly with $S$. | THz near-field, on-chip RF, qubits, bio-EM |
| G3 | **No robust GPU preconditioner for indefinite complex Maxwell problems with PML.** | Commercial frequency-domain GPU paths rely on direct solvers or DDM with data-centre FP64 GPUs (CST requires B200 or H100 for DDM). | Every frequency-domain user |
| G4 | **No FP32-first solvers with rigorous error control for consumer GPUs.** | Palace documents no single-precision option. Consumer and now B300 GPUs run FP64 at 1:64. | Most researchers, who have workstation or laptop GPUs |
| G5 | **VRAM limits make direct solvers unusable** at 0.3M+ DOFs on 8 to 16 GB cards. | Measured 6 GB on CPU for 289k DOFs (MUMPS). A degree-2 run ran out of memory at 0.42M DOFs on 7 GB. | Laptop and workstation users |
| G6 | **No open GPU DGTD for Maxwell with LTS, curved tets and dispersive metals.** | Only research codes found. Even SeisSol's GPU LTS efficiency is mediocre. | Optical plasmonics and metasurfaces with moderate $S$ |
| G7 | **Scan workflows produce mesh noise.** | CST and the prototype both remesh per tip position. Point values scatter by about ±10 %. | All scanning-probe simulations, parameter sweeps, optimization |
| G8 | **No benchmark suite for multi-scale near-field problems.** | No public reference results with error bars for tip-sample coupling at high $S$. | Code developers and validation |
| G9 | **Usability.** | Geometry, meshing, materials and post-processing remain far behind CST and HFSS in open source. | Adoption by experimentalists |

The most valuable and most tractable target is **G1 + G2 + G5 for time domain**, which the implicit FETD route closes. G3 and G4 for frequency domain are research problems that can come later, and partly sidestepped by EM-WaveHoltz.

---

## 5. Technical analysis for the bowtie problem

All numbers below are estimates for one scan point with broadband output (0.2 to 2 THz), except where marked as measured. They use the prototype mesh (236k curved tets, 289k first-order N1curl DOFs, element size 10 nm at the apex).

### 5.1 Explicit FDTD or FIT (measured in CST)

- 1.6M cells with 20 nm local cells and 100 nm gaps, 84,009 steps at 11.5 ms on 8 CPU threads, about 18 min.
- With 30 nm gaps and 5 nm cells the step drops about 4×, which made the scan infeasible.
- A GPU would give 5 to 20× in CST, but staircasing and mesh-dependent apex fields remain.

### 5.2 Explicit DGTD with local time stepping (estimate)

The estimate assumes $p = 3$ on curved tets, $N_p = 20$ nodes per element, six field components and five-stage low-storage Runge-Kutta.

- **Work per element and stage.** Volume terms about $3 \times 6 \times 2 N_p^2 \approx 1.4\times10^4$ flops, surface lift about $6 \times 2 \times N_p \times 4 N_{fp} \approx 10^4$ flops, total about $2.5\times10^4$ flops, or about $1.2\times10^5$ flops per element and step.
- **Finest time step.** With $h_{\min} \approx 15$ nm (a $p = 3$ curved element resolves a 50 nm sphere at this size), $r_\text{in} \approx 3$ nm and $\Delta t \approx 0.5 \cdot 3\,\text{nm}/(c \cdot 16) \approx 3\times10^{-19}$ s.
- **Simulated time.** About 15 ps (pulse, ringing, echoes in a 50 µm Si slab), so about $5\times10^7$ steps for the finest cluster. A semi-infinite substrate in the PML would cut this to 5 to 8 ps.
- **Fine elements.** A graded point refinement with growth rate $g$ has roughly $8/g^3 \times$ (shape factor) elements per factor-of-two size band, about $4\times10^3$ for $g = 0.25$. With rate-2 clustering the total work is about twice the finest cluster's.
- **Total.** About $2 \times 4\times10^3 \times 5\times10^7 \times 1.2\times10^5 \approx 5\times10^{16}$ flops.
- **Time.** At 3 to 5 TFLOP/s sustained FP32 on the laptop that is 3 to 5 hours per point. On an H100-class GPU (about 30 TFLOP/s sustained) it is about 30 min.

DGTD with LTS works and is elegant, but at $S \sim 10^4$ it is not competitive on modest hardware. Its cost grows linearly with $S$, so it is the right tool for photonics with $S \le 10^3$, where the same estimate gives seconds to minutes.

### 5.3 Implicit FETD with AMS-preconditioned CG (estimate)

- **Time step.** Generalized-α or Newmark with $\Delta t \approx 20$ to 25 fs (20 steps per period at 2 THz, 45 at 0.9 THz). Over 15 ps that is about 600 to 750 steps.
- **Matrix size.** About $2.9\times10^5$ DOFs and about 25 nonzeros per row, so about $7\times10^6$ nonzeros, about 90 MB in FP64 CSR.
- **SpMV.** About 0.35 ms at 256 GB/s.
- **AMS V-cycle.** About 10 to 20 SpMV equivalents (nodal vector Poisson and gradient correction), so 4 to 7 ms.
- **CG iterations per step.** With a predictor from the previous two steps, about 5 to 15 to reach $10^{-8}$, so about 50 to 100 ms per step.
- **Total per point on the laptop GPU.** About 30 to 75 s, plus a few seconds for assembly and AMS setup. Call it **1 to 2 min**.
- **Same code on the laptop CPU.** About 3× slower (bandwidth ratio), so 3 to 5 min, still better than the frequency sweep.
- **On a data-centre GPU.** 10 to 30× faster than the laptop GPU, a few seconds per point.
- **Memory.** Matrix, AMS hierarchy and vectors take about 0.5 to 1 GB, which fits easily in 8 GB, so two to four scan points can run concurrently, or as a block of right-hand sides.

The iteration counts are the main uncertainty. AMS is robust for these SPD systems, but the PML and conductivity terms can raise the counts somewhat. A prototype must measure this.

### 5.4 Frequency-domain FEM, CPU direct (measured and extrapolated)

- 45 s per frequency on 2 slow cores (measured). About 15 to 25 s on the laptop with 6 ranks.
- 20 to 30 frequencies are needed because of the sharp resonance between 0.9 and 1.2 THz, so 7 to 10 min per point.
- Fast-sweep model order reduction could cut this to 5 to 8 full solves, about 2 to 3 min.
- It does not map well to the laptop GPU (FP64 compute-bound, VRAM).

### 5.5 Summary

| Approach | Per point, laptop | Per point, data-centre GPU | Mesh quality | Implementation effort |
|---|---|---|---|---|
| CST FIT (hex), measured | about 18 min (CPU) | a few min | Staircased | None |
| Explicit DGTD + LTS | 3 to 5 h | about 30 min | Curved, excellent | High |
| **Implicit FETD + AMS-CG** | **1 to 2 min (GPU), 3 to 5 min (CPU)** | **seconds** | **Curved, excellent** | **Medium** |
| Frequency-domain FEM, direct | 7 to 10 min (CPU) | limited by VRAM | Curved, excellent | Done (prototype) |
| Frequency-domain FEM + fast sweep | 2 to 3 min (CPU) | limited by VRAM | Curved, excellent | Medium |

For the full 9,800-point scan, the implicit FETD route means about 7 to 14 days on the laptop GPU without thinning. An adaptive scan grid (dense near the gap and edges) should reduce the point count by 5 to 10×, which gives 1 to 3 days.

---

## 6. Recommended solver architecture (Track A, implicit FETD)

### 6.1 Formulation

- **Unknown.** Total or scattered electric field $\mathbf{E}$ in the second-order vector wave equation
  $$\varepsilon\,\partial_t^2 \mathbf{E} + \sigma\,\partial_t \mathbf{E} + \nabla\times\mu^{-1}\nabla\times\mathbf{E} = -\partial_t \mathbf{J}.$$
- **Space.** Nédélec first kind, order 1 or 2, on second-order curved tetrahedra.
- **Time integration.** Newmark with β = 1/4 (average acceleration) or generalized-α with ρ∞ ≤ 1. Both are second order and unconditionally stable, and some damping of the highest modes (ρ∞ < 1) is useful against PML and mesh-induced spurious modes.
- **System per step.** $A = M_\varepsilon + \gamma\Delta t\,C_\sigma + \beta\Delta t^2 K$, real and SPD when the PML memory terms go to the right-hand side.

### 6.2 Open boundaries

- **Option 1 (recommended).** The anisotropic, dispersive PML formulation of Jiao, Jin, Michielssen and Riley (IEEE TAP 2003). It works for total- and scattered-field formulations, lets the PML parameters vary inside elements (needed for higher-order bases), and admits unconditionally stable schemes. Implement the auxiliary variables as element-local ODEs updated explicitly, and keep the stiffness part on the left-hand side so $A$ stays SPD. Check that the chosen splitting preserves stability, since this is where most FETD-PML implementations go wrong.
- **Option 2.** A complex-frequency-shifted (CFS) stretching with recursive convolution. It absorbs evanescent and grazing waves better, which matters for the Si slab running into the PML.
- **Fallback.** First-order absorbing boundary (Palace has it in time domain) with a larger domain. This is simple but reflects about 1 to 10 % at oblique incidence, so it is unsuitable for quantitative enhancement maps.

### 6.3 Excitation

- **Scattered-field formulation**, as in the prototype. The background is the analytic plane wave on the layered substrate in the time domain, obtained by inverse FFT of the frequency-domain slab solution times the pulse spectrum (pre-tabulated as a function of $\mathbf{k}_t\cdot\mathbf{r}$, $z$ and $t$).
- **Boundary data.** On PEC surfaces, $\mathbf{n}\times\mathbf{E}_s = -\mathbf{n}\times\mathbf{E}_\text{inc}$. Inside penetrable scatterers, a volume source term $-(\varepsilon - \varepsilon_b)\,\partial_t^2\mathbf{E}_\text{inc}$.
- **Shaft tapering.** Keep the window on the incident field along the upper shaft (`--win` in the prototype), which models a finite beam on a mm-long tip.
- **Alternative.** A Huygens surface (TF/SF) around the antenna region.

### 6.4 Materials

- **Au at THz.** PEC agrees within 1 % in the prototype, so it is the default. For higher accuracy, a time-domain surface impedance through vector fitting of $Z_s(\omega) \propto \sqrt{-i\omega}$ and recursive convolution (a few auxiliary variables per boundary DOF), or a Drude volume model with an auxiliary differential equation, which needs skin-depth resolution.
- **Doped or photoexcited Si.** Drude ADE (adds a damping term and one auxiliary current).
- **2D materials.** Graphene or TMD sheets as surface conductivity on interior faces, Drude form for THz.

### 6.5 Linear solver

- **CG with AMS** in FP64 (outer), with the option of applying the AMS V-cycle in FP32 (flexible CG or a few inner iterations). The coarse auxiliary problems are SPD scalar and vector Poisson, which suit GPU AMG.
- **Warm start and predictor.** $e^{n+1}_0 = 2e^n - e^{n-1}$, or a projection onto the last 5 to 10 solutions, which can halve the iteration count.
- **For order 2 and above.** p-multigrid with Chebyshev smoothing and partial assembly on fine levels, with AMS on the order-1 level (as in Palace).
- **Many right-hand sides.** For several scan points sharing one mesh (see 6.7), block CG or pipelined batching amortizes the bandwidth.

### 6.6 Outputs

- On-the-fly DFT at a set of frequencies for the probe point clouds, the gap line integral and optional planar near-field cuts.
- Time traces at the probe, for direct comparison with THz-STM and EOS experiments.
- Normalization by the incident pulse spectrum gives the enhancement spectrum.

### 6.7 Scan workflow and mesh consistency (closing G7)

This is where an open solver can do something CST cannot.

1. **Remesh with a tip-anchored size field** (prototype). It is simple and reproducible, but the discretization error jumps a little between points.
2. **Mesh morphing (recommended).** Build one mesh with a refinement "bubble" around the tip. For each scan position, move the nodes inside a buffer region (harmonic or elasticity-based displacement) so the tip translates and the topology stays fixed. The sparsity pattern, the AMS setup structure and the error character are then continuous in tip position, so **the enhancement map becomes a smooth function of position with no remeshing noise.** Remesh only when element quality drops below a threshold, and overlap a few points across each remesh to detect jumps.
3. **Non-conforming tip subdomain.** A separate tip mesh coupled to the antenna mesh across a sliding interface with mortar or interior-penalty coupling, as in rotating-machine simulation. This is the most flexible but the most complex option.

### 6.8 Precision strategy

- Assembly in FP64 (it is a one-time cost).
- Matrix storage FP64, or FP32 with FP64 accumulation if bandwidth is the limit. Measure the effect on CG convergence.
- AMS hierarchy in FP32, used as a preconditioner inside FP64 flexible CG.
- Vectors and dot products in FP64. They are memory-bound and cheap even at 1:64.
- Verify with an FP64-everywhere reference run on every new problem class.

### 6.9 Verification and validation suite (also closing G8)

1. PEC and dielectric sphere under plane wave versus Mie series, time and frequency domain.
2. Dipole above a dielectric half-space versus a Sommerfeld-integral solution, to test the layered background and the PML at grazing angles.
3. PML reflection versus angle and frequency, with the substrate penetrating the PML.
4. Sphere-plane gap (nanogap) versus the quasi-static multipole solution at small gaps, a direct analogue of the tip-sample junction.
5. Bowtie without tip versus the FEniCSx frequency-domain prototype and CST.
6. Full bowtie with tip, convergence in $h$, $p$, $\Delta t$, domain size and PML thickness, with error bars on the enhancement.
7. Publish the geometry, meshes and reference results as an open benchmark.

---

## 7. Other tracks

### 7.1 Track B, explicit DGTD with clustered LTS (for $S \le 10^3$)

- Nodal DG, order 2 to 4, upwind flux, curved tets, low-storage Runge-Kutta or ADER.
- Rate-2 clustered LTS with 3 to 6 clusters, following the EDGE scheme (neighbors restricted to adjacent clusters, about 1.5 % loss).
- FP32 throughout, with batched small matrix products (tensor cores where shapes allow).
- CFS-PML, TF/SF, Drude-Lorentz ADEs, critical-point models for Au and Ag in the visible.
- The main engineering problem is GPU load balance across clusters of very different size, which SeisSol has not fully solved. Options include separate kernel streams per cluster, cluster-aware partitioning and merging small clusters.
- This fills G6 for optical plasmonics and metasurfaces, and a vendor-neutral implementation (Julia KernelAbstractions, JAX, Kokkos or SYCL) would reach the widest audience. It is a larger effort than Track A and does not serve the THz problem.

### 7.2 Track C, GPU frequency-domain solver (research)

- **Optimized Schwarz DD** with impedance transmission conditions. The subdomain problems are solved by batched GPU direct solvers in FP32 (cuDSS with downcast factors) inside FP64 GMRES. Add a GenEO-type coarse space.
- **EM-WaveHoltz.** Reuse the Track A time stepper to solve single-frequency problems through a positive definite iteration. This is attractive because it needs no new preconditioner.
- **Fast sweeps.** Model order reduction or AAA on top of either approach.
- This attacks G3 and G4 and is publishable research, but it is not needed for the bowtie scan.

### 7.3 Track D, quasi-static near zone coupled to full wave

- One full-wave solve without the tip (or with the tip at a reference position), then at each scan position a fast electrostatic tip-sample solve (scalar Poisson, about 10⁵ DOFs, seconds on GPU with AMG).
- Valid for the apex region (100 nm ≪ λ) but not for the shaft, whose antenna response is about λ/3. Use it only after calibrating against 5 to 10 full-wave points.
- This is the cheapest option for very dense scans and a useful cross-check for Track A.

---

## 8. Implementation paths

| Path | Pros | Cons | Fit |
|---|---|---|---|
| **1. Prototype in FEniCSx (real PETSc + hypre)** | Reuses the existing gmsh geometry, background field and readouts. Python speed of development. AMS works because the time-domain problem is real. | GPU path needs PETSc built with CUDA. Assembly stays on the CPU unless cuDOLFINx catches up with the dolfinx version. | Fastest route to measured iteration counts and timings |
| **2. Contribute to Palace** | Production C++ code with GPU, AMS, p-multigrid, curved high-order elements, implicit transient and AWS maintenance. Community impact. | Larger codebase, upstream review, design discussions on PML and excitation. | Best route to community value |
| 3. Own driver on MFEM | Full control, all MFEM features | Rebuilds much of Palace | Only if Palace's design blocks needed features |
| 4. New code (Julia or JAX) | Vendor-neutral, differentiable | Large effort, reinventing FEM infrastructure | Only for Track B |

**Recommendation.** Path 1 to answer the technical questions within weeks, then Path 2 to deliver the missing pieces (time-domain PML, scattered-field plane-wave excitation with layered backgrounds, time-domain surface impedance, single-precision preconditioner option, DFT probes) to a code that many groups already use. Open an issue with the Palace maintainers early, since PML is already on their roadmap.

---

## 9. Roadmap

Durations are rough and assume one person working part-time.

| Phase | Content | Exit criterion |
|---|---|---|
| 0. Baseline (1 week) | Run `run_convergence.sh` on the laptop. Record frequency-domain cost and converged enhancement. | Reference enhancement spectrum with error bars at 3 tip positions |
| 1. FETD prototype on CPU (3 to 4 weeks) | FEniCSx real build, Newmark, first-order ABC, scattered-field excitation with tapered shaft, AMS-CG, DFT probes | Agrees with the frequency-domain baseline within 2 %, measured iterations per step |
| 2. PML (3 to 4 weeks) | Jiao-Jin or CFS-PML with auxiliary ODEs, stability tests | Reflection below 10⁻³ over 0.2 to 2 THz and 0 to 70°, long runs stable |
| 3. GPU (2 to 3 weeks) | PETSc CUDA plus hypre on device, timing, FP32 preconditioner option | Measured time per point, compared with the estimate of 1 to 2 min |
| 4. Scan workflow (3 to 4 weeks) | Mesh morphing, adaptive scan grid, batching | Smooth enhancement map, no remeshing jumps above the convergence error |
| 5. Upstream to Palace (2 to 4 months) | PML, plane-wave and layered background, surface impedance, probes, benchmark suite | Merged features, published benchmark |
| 6. Optional research | EM-WaveHoltz on the FETD kernel, Track B or C | Publication |

**Go / no-go after phase 3.** If AMS needs more than about 30 iterations per step with the PML, or the GPU gain is below 2× over the CPU, reconsider. Fall back to the CPU frequency domain with a fast sweep, or to Track D for dense scans.

---

## 10. Risks and open questions

1. **PML stability in implicit FETD.** Splitting the PML terms explicitly can reintroduce a stability limit. Jiao et al. show unconditionally stable variants exist, but the chosen implementation must be checked with long runs.
2. **AMS iteration growth.** PML damping, high contrast (Si ε = 11.7 against air) and very anisotropic elements near edges may raise counts. Measure, and consider p-multigrid or deflation if needed.
3. **Spurious DC modes.** Implicit curl-curl schemes can accumulate static (gradient) modes from incompatible sources or roundoff. Use divergence-free source construction, a small ρ∞ < 1 damping, or a periodic gradient projection.
4. **Background field accuracy in time domain.** Inverse FFT of the slab solution must handle the Si echoes and causality. Window and pad the pulse carefully.
5. **Mesh morphing quality.** Large tip displacements degrade elements. Choose the buffer size and remesh triggers by test.
6. **Software stack on the laptop.** WSL2 with CUDA works, but PETSc and hypre with CUDA must be built from source. Pin versions.
7. **Upstream acceptance.** Palace's maintainers may already have a PML design. Coordinate before implementing.
8. **Estimate risk.** The 1 to 2 min per point figure is an estimate. Phases 1 and 3 exist to replace it with measurements.

---

## 11. Why this matters beyond the bowtie

- **Experimental near-field microscopy** (THz-STM, THz and IR s-SNOM, tip-enhanced spectroscopy) needs quantitative, mesh-independent tip-sample enhancement for data interpretation. Today groups rely on commercial licenses or quasi-static models.
- **On-chip THz and RF** (antennas, photoconductive emitters, qubit packaging) shares the same high-$S$ regime.
- **2D materials photonics** needs thin conductive sheets inside large domains, which surface conductivity on curved tet meshes handles naturally.
- **Inverse design and optimization** need fast, smooth, differentiable solves. An implicit FETD with a fixed-topology mesh is a natural basis for adjoint gradients.

---

## References

**Tools and documentation**
- Palace documentation (features, solver configuration, boundary conditions, GPU). https://awslabs.github.io/palace/stable , https://awslabs.github.io/palace/dev/config/solver , https://awslabs.github.io/palace/dev/guide/boundaries/ , https://awslabs.github.io/palace/v0.16.0/guide/parallelism/
- CST Studio Suite GPU Computing Guide 2026. https://updates.cst.com/downloads/GPU_Computing_Guide.pdf
- Ansys 2025 R1 GPU accelerator capabilities. https://ansys.com/content/dam/release/2025-r1/platform-support/ansys-2025-r1-gpu-accelerator-capabilities.pdf
- COMSOL, "Faster Simulation with NVIDIA GPU Support for COMSOL Multiphysics". https://comsol.com/blogs/faster-simulation-with-nvidia-gpu-support-for-comsolmph
- NVIDIA cuDSS documentation and data types. https://docs.nvidia.com/cuda/cudss/latest/types.html
- cuda-dolfinx (cuDOLFINx). https://github.com/bpachev/cuda-dolfinx
- hypre GPU AMG, Exascale Computing Project highlight. https://www.exascaleproject.org/highlight/a-new-approach-in-the-hypre-library-brings-performant-gpu-based-algebraic-multigrid-to-exascale-supercomputers-and-the-general-hpc-community/
- Survey of open-source FDTD and RCWA solvers. https://geniuskey.github.io/compass/research/open-source-em-solvers-survey.html
- FDTDX. https://arxiv.org/abs/2412.12360
- CUTh-Solver, GPU sparse solver comparison for 3D IC thermal simulation, 2026. https://arxiv.org/pdf/2606.17850

**Methods**
- D. Jiao, J.-M. Jin, E. Michielssen, D. J. Riley, "Time-domain finite-element simulation of three-dimensional scattering and radiation problems using perfectly matched layers", IEEE Trans. Antennas Propag., 2003.
- Z. Peng, D. Appelö, "EM-WaveHoltz: A flexible frequency-domain method built from time-domain solvers", IEEE Trans. Antennas Propag., 2022. https://arxiv.org/abs/2103.14789
- R. Hiptmair, J. Xu, "Nodal auxiliary space preconditioning in H(curl) and H(div) spaces", SIAM J. Numer. Anal., 2007.
- Tz. Kolev, P. Vassilevski, "Parallel auxiliary space AMG for H(curl) problems", J. Comput. Math., 2009.
- V. Dolean, M. J. Gander, L. Gerardo-Giorda, "Optimized Schwarz methods for Maxwell's equations", SIAM J. Sci. Comput., 2009.
- N. Bootland, V. Dolean, I. G. Graham, C. Ma, R. Scheichl, E. A. Spence, "Overlapping Schwarz methods with GenEO coarse spaces for indefinite and non-self-adjoint problems", IMA J. Numer. Anal., 2023.
- P. Tsuji, B. Engquist, L. Ying, "A sweeping preconditioner for time-harmonic Maxwell's equations with finite elements", J. Comput. Phys., 2012.
- E. Carson, N. J. Higham, "Accelerating the solution of linear systems by iterative refinement in three precisions", SIAM J. Sci. Comput., 2018.
- Y. Nakatsukasa, O. Sète, L. N. Trefethen, "The AAA algorithm for rational approximation", SIAM J. Sci. Comput., 2018.
- J. S. Hesthaven, T. Warburton, *Nodal Discontinuous Galerkin Methods*, Springer, 2008.
- A. Klöckner, T. Warburton, J. Bridge, J. S. Hesthaven, "Nodal discontinuous Galerkin methods on graphics processors", J. Comput. Phys., 2009.
- N. Gödel, S. Schomann, T. Warburton, M. Clemens, "GPU accelerated Adams-Bashforth multirate discontinuous Galerkin FEM simulation of high-frequency electromagnetic fields", IEEE Trans. Magn., 2010.
- M. J. Lizarazo, E. J. Silva, "A parallel-GPU DGTD algorithm with a third-order LTS scheme for solving multi-scale electromagnetic problems", Mathematics, 2024. https://www.mdpi.com/2227-7390/12/23/3663
- K. Busch, M. König, J. Niegemann, "Discontinuous Galerkin methods in nanophotonics", Laser Photonics Rev., 2011.
- A. Breuer, A. Heinecke, "Next-generation local time stepping for the ADER-DG finite element method". https://arxiv.org/abs/2202.10313

**Hardware**
- "Nvidia's Blackwell Ultra upends 15 years of GPU segmentation strategy" (B300 FP64 and the Ozaki scheme). https://news.lavx.hu/article/nvidia-s-blackwell-ultra-upends-15-years-of-gpu-segmentation-strategy
- S. Matsuoka, "FP8 is All You Need (Part 1): Debunking Hardware FP64 as the HPC Holy Grail", arXiv:2606.06510, 2026 (analytical projections). https://arxiv.org/html/2606.06510
