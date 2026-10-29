# Track C in detail. A GPU frequency-domain Maxwell solver for multi-scale problems

*Companion to `gpu_em_solver_landscape.md`. Written October 2026. The reference problem is the THz bowtie antenna with a near-field probe tip. Numbers marked "measured" come from experiments run for this document on the FEniCSx prototype matrices (scripts in `fenicsx/trackc/`). Everything else marked as an estimate needs confirmation on real GPU hardware.*

---

## 0. Summary

- **Track C was framed as a research problem**, robust GPU solvers for indefinite, PML-loaded, frequency-domain Maxwell systems. For the bowtie problem the experiments below show that the dominant difficulty is a different one, and that it has a cheap, known fix.
- **Two separate sources of ill-conditioning.** (1) Wave indefiniteness, which grows with the number of wavelengths across the domain. Here the domain is only about 1.5 wavelengths across in air and about 5 in the Si, so this part is mild. (2) Low-frequency breakdown, which grows with the scale separation $S = \lambda/h_{\min}$. The discrete gradient fields form a near-null space, and the effective condition number grows like $(S/2\pi)^2 \approx 10^6$ to $10^8$. This part dominates.
- **Measured.** FP32 sparse factorization on its own is useless for this matrix. The residual looks fine (10⁻³ to 10⁻⁶), but the solution error is 16 % to 8,500 %, and classical iterative refinement diverges in every case tested.
- **Measured.** Adding a gradient-space correction (a nodal problem about 6.7× smaller, the mechanism behind AMS) makes FP32 factorization work. Refinement converges to a relative residual of 10⁻¹² in 3 to 8 steps on 65k, 138k and 289k DOF meshes from 0.3 to 2 THz. The solution matches the FP64 solution to 10⁻⁷ to 10⁻¹⁰, which is within the FP64 reference's own uncertainty. The correction works equally well in FP32.
- **Measured.** The FP32 factor of the 289k DOF production mesh needs 1.3 GB (2.6 GB in FP64). **It fits in your 8 GB GPU with room to spare.** This corrects my earlier statement in our conversation that the factorization would not fit.
- **Measured.** One FP32 factorization preconditions nearby frequencies. At ±5 % it needs 19 GMRES iterations, at ±11 % about 30, at ±33 % 90 to 140, all reaching FP64 accuracy.
- **Recommended design (C1).** FP32 sparse direct factorization on the GPU (cuDSS) plus a nodal gradient correction plus FP64 residuals in a refinement loop. Add frequency recycling, model order reduction for the sweep, affine frequency assembly, and mesh morphing with reuse of the symbolic analysis across scan positions.
- **Estimated payoff.** About 0.5 to 1.5 s per frequency and 5 to 15 s per broadband scan point on the RTX 2000 Ada. That is faster than the Track A estimate (1 to 2 min) for this problem size, and it keeps the exact frequency-domain physics (surface impedance, dispersion, PML).
- **Even without a GPU.** The same algorithm on the laptop CPU (FP32 factors, recycling, adaptive sweep) should cut the current 7 to 10 min per scan point to about 0.5 to 1.5 min. Most of the gain is algorithmic, and the GPU adds roughly another 5×.
- **Where C1 stops.** Above roughly 1M DOFs on an 8 GB card, or for domains many wavelengths across, factor memory and wave indefiniteness return. The scale-up variants are two-level domain decomposition with C1 as the subdomain solver (C2), matrix-free complex-shifted auxiliary-space multigrid (C3), and EM-WaveHoltz on the Track A time stepper (C4). Fu (2026) recently demonstrated a C3-type solver at 10¹⁰ unknowns on 64 GPUs, in the opposite regime (many wavelengths, uniform resolution). Section 6.2.1 compares it with this problem.
- **First real step.** Benchmark cuDSS on the exported matrices on your laptop. That one measurement decides whether C1 is the main path.

---

## 1. Why a frequency-domain GPU solver at all

Track A (implicit time domain) covers broadband scans well. A frequency-domain solver is still worth having for these reasons.

| Need | Frequency domain | Time domain |
|---|---|---|
| Dispersive materials (Drude, Lorentz, measured $\varepsilon(\omega)$ tables) | Exact, just a complex number per frequency | Needs auxiliary equations or vector fitting |
| Surface impedance $Z_s \propto \sqrt{\omega}$ | Exact | Needs recursive convolution |
| 2D sheet conductivity (graphene, TMDs) $\sigma(\omega)$ | Exact | Needs fitting |
| Narrowband or CW experiments, lock-in at one frequency | One solve | Full pulse run |
| High-Q resonances | Direct | Long ring-down |
| Eigenmodes and resonance tracking | Natural (shift-invert) | Indirect |
| Adjoint gradients for inverse design at fixed frequencies | One extra solve with the same factorization | Time-reversed run with stored history |
| PML | Exact complex stretching | Auxiliary fields, stability care |

For the bowtie scan the deciding factors are speed per scan point and the exactness of the Au surface impedance and of the PML.

---

## 2. The linear systems

### 2.1 Formulation

With $e^{-i\omega t}$, $k_0 = \omega/c$, scattered-field unknown $\mathbf{E}_s$, Nédélec (N1curl) elements of order $p$ on curved tetrahedra, the prototype assembles

$$A(\omega) = K_\Lambda - k_0^2 M_{\varepsilon\Lambda} - \frac{i k_0}{z_s(\omega)} B_{Au},$$

- $K_\Lambda$ is the curl-curl matrix with the PML tensor $\Lambda^{-1}$,
- $M_{\varepsilon\Lambda}$ is the mass matrix weighted with $\varepsilon_r \Lambda$,
- $B_{Au}$ is the tangential mass on the Au boundary (Leontovich condition, $z_s = Z_s/\eta_0$),
- Dirichlet rows (PEC tip, outer boundary) are identity rows after symmetric lifting.

**Measured.** The assembled matrices are complex symmetric, $A = A^T$ to $2\times10^{-13}$, not Hermitian. That allows an $LDL^T$ factorization with half the storage of $LU$.

### 2.2 The matrix set

| Mesh | Element size at apex and gap | DOFs | nnz(A) | Vertices (gradient space) |
|---|---|---|---|---|
| c65k | 30 nm | 64,930 | 1.0 M | 10,255 |
| f138k | 10 nm | 137,865 | 2.18 M | 20,642 |
| d289k (production default) | 10 nm, finer far mesh | 289,444 | 4.60 M | 43,043 |

All three use first-order N1curl on second-order curved tets. They come from the same script as the prototype runs, so the physics and readouts are identical.

### 2.3 Two kinds of ill-conditioning

**Wave indefiniteness.** At 2 THz the domain including PML is about 230 µm wide, about 1.5 free-space wavelengths and about 5 wavelengths in Si. The number of negative eigenvalues of $A$ and the pollution effect scale with this $kL$, which is small here. Large-domain Helmholtz and Maxwell solvers (sweeping, optimized Schwarz) are built for $kL$ of 10² to 10³. They are not needed at this size.

**Low-frequency breakdown.** The N1curl space contains the discrete gradients $G\phi$ ($G$ is the vertex-to-edge incidence matrix, $\phi$ nodal values). Since $\nabla\times\nabla\phi = 0$,

$$A\,G\phi = -k_0^2 M_{\varepsilon\Lambda} G\phi + (\text{boundary terms}).$$

On these fields $A$ behaves like $k_0^2 \times$ a mass matrix, while on the curl-active fields it behaves like $1/h^2$ (per unit mass). After diagonal scaling, the ratio over the mesh is about

$$\kappa_\text{eff} \sim \frac{1}{(k_0 h_{\min})^2} = \left(\frac{\lambda}{2\pi h_{\min}}\right)^2 = \left(\frac{S}{2\pi}\right)^2 .$$

| Frequency | λ | $S$ at $h_{\min} = 30$ nm | $\kappa_\text{eff}$ |
|---|---|---|---|
| 0.3 THz | 1000 µm | 3.3×10⁴ | 3×10⁷ |
| 0.9 THz | 333 µm | 1.1×10⁴ | 3×10⁶ |
| 2.0 THz | 150 µm | 5×10³ | 6×10⁵ |

With $h_{\min} = 10$ nm these numbers grow ninefold. FP32 has unit roundoff $u_{32} \approx 6\times10^{-8}$, so $\kappa u_{32}$ ranges from about 0.04 to well above 1. **Plain FP32 factorization is expected to fail at exactly the high scale separations that motivated this project.** This is the same physics as the well-known low-frequency breakdown of EM integral and FEM formulations, and it is why AMS splits the problem into gradient and curl parts.

The cure follows from the structure. The bad subspace is known exactly and algebraically (it is range($G$)), and it is small (one unknown per vertex, about 15 % of the edge count). Solving it separately removes the bad part of the spectrum. The rest is well conditioned enough for FP32.

---

## 3. Measured results

All runs used MUMPS through python-mumps on a 2-core cloud CPU, sequential, complex symmetric $LDL^T$, default ordering and scaling. CPU MUMPS stands in for cuDSS here. The goal was to measure numerical behaviour and memory, not GPU speed. "Error" is the relative 2-norm difference to the PETSc/MUMPS FP64 solution of the prototype. That reference itself differs from an independent FP64 factorization by 10⁻⁸ to 10⁻¹⁰, consistent with $\kappa u_{64}$, so errors at that level mean "as accurate as FP64".

### 3.1 Factorization cost and memory (FP64 versus FP32)

| Mesh | nnz(L) | MUMPS operation count | Memory FP64 | Memory FP32 | Factor time FP64 | Factor time FP32 |
|---|---|---|---|---|---|---|
| c65k | 14.1 M | 8.0×10⁹ | 347 MB | 180 MB | 0.92 s | 0.64 s |
| f138k | 44.2 M | 4.4×10¹⁰ | 1,020 MB | 524 MB | 3.4 s | 2.2 s |
| d289k | 113 M | 1.6×10¹¹ | 2,582 MB | 1,319 MB | 10.5 s | 6.3 s |

- Fill grows like about $N^{1.4}$ and operations like about $N^{2}$ over this range (rough, since the three meshes are graded differently). Extrapolated to 1M DOFs, that is about 640M factor entries (about 5 GB in complex FP32) and about 2×10¹² operations.
- On a CPU, FP32 is only 1.5 to 1.7× faster. On a consumer GPU the FP32 to FP64 throughput ratio is 64, so precision decides whether the GPU is useful at all.

### 3.2 Plain FP32 factorization fails

| Matrix | Residual of FP32 solve | Error of FP32 solve | Classical refinement (LU-IR) | GMRES-IR (FP64 GMRES, 600 iterations) |
|---|---|---|---|---|
| c65k, 0.3 THz | 5.7×10⁻⁴ | 1000 % | diverges | stalls, residual 2×10⁻⁵ |
| c65k, 0.9 THz | 7.8×10⁻⁵ | 280 % | diverges | stalls, residual 7×10⁻⁸ |
| c65k, 1.2 THz | 4.7×10⁻⁵ | 128 % | diverges | converges in 523 iterations |
| c65k, 2.0 THz | 6.0×10⁻⁶ | 16 % | diverges | converges in 193 iterations |
| f138k, 0.9 THz | 6.1×10⁻³ | 8,500 % | diverges | stalls, residual 2×10⁻⁴ |
| d289k, 0.9 THz | 2.1×10⁻³ | 2,800 % | diverges | stalls, residual 8×10⁻⁴ |

- The error falls about 64× from 0.3 to 2.0 THz, close to the 44× predicted by $\kappa \propto \lambda^2$ in Section 2.3.
- The small residuals are deceptive. A residual check alone would accept these wrong solutions.
- GMRES-based refinement (Carson and Higham, Amestoy et al.) is more robust than LU-IR in theory, but here it needs hundreds of iterations or stalls. The condition number is beyond what it can absorb.

### 3.3 FP32 factorization with gradient correction works

Preconditioner (symmetric multiplicative subspace correction).

```
P(r):  y  = G S⁻¹ Gᵀ r                  # gradient correction, S = Gᵀ A G (nodal)
       y += L32⁻¹ (r − A y)             # FP32 LDLᵀ solve of the full edge system
       y += G S⁻¹ Gᵀ (r − A y)          # gradient correction again
Refinement:  x ← x + P(b − A x)   with b − A x computed in FP64
```

$S = G^T A G$ is computed algebraically, so it automatically contains the PML, the Si permittivity, the surface impedance and the boundary rows. It is complex symmetric, with one unknown per vertex.

| Matrix | Steps to residual 10⁻¹² (S in FP64) | Error | Steps (S in FP32) | Error |
|---|---|---|---|---|
| c65k, 0.3 THz | 8 | 1.4×10⁻⁸ | 8 | 1.2×10⁻⁸ |
| c65k, 0.9 THz | 5 | 1.5×10⁻⁹ | 5 | 1.5×10⁻⁹ |
| c65k, 2.0 THz | 3 | 1.5×10⁻¹⁰ | 3 | 1.6×10⁻¹⁰ |
| f138k, 0.3 THz | 8 | 1.0×10⁻⁷ | 8 | 1.0×10⁻⁷ |
| f138k, 0.9 THz | 8 | 1.1×10⁻⁸ | 8 | 1.1×10⁻⁸ |
| f138k, 2.0 THz | 6 | 1.3×10⁻⁹ | 6 | 1.3×10⁻⁹ |
| d289k, 0.9 THz | 7 | 9.2×10⁻⁹ | 7 | 9.1×10⁻⁹ |

- **Robust.** The step count does not grow with mesh refinement (65k to 289k) or with scale separation (0.3 THz on the 10 nm mesh is the hardest case, $S \approx 10^5$).
- **Fully FP32 except residuals.** The nodal solve can also be FP32. Only the residual $b - Ax$ and the update need FP64, and both are memory-bound vector operations that run at full speed on a consumer GPU.
- **Cheap.** $S$ is 6.7× smaller than $A$ and its factor is a small fraction of the edge factor. On the CPU the 7 refinement steps on d289k took 3.5 s (FP64 $S$) against 6.3 s for the FP32 factorization itself. On a GPU they are dominated by triangular solves, which are bandwidth-bound.

*Note on the GMRES columns in the raw output.* SciPy's GMRES with left preconditioning stops on the preconditioned residual, which hides the true residual. The tables use the stationary refinement, which checks the true FP64 residual every step. The recycling test below uses right preconditioning with an outer refinement loop for the same reason.

### 3.4 Reusing one factorization across frequencies

Factorization (FP32) at 0.9 THz, gradient correction rebuilt for the target frequency (cheap), GMRES-IR (outer refinement on the true FP64 residual, inner right-preconditioned GMRES to 10⁻⁶), stopped at a residual of 10⁻¹¹. Mesh c65k.

| Target frequency | Shift from 0.9 THz | Outer steps | Total inner GMRES iterations | Error |
|---|---|---|---|---|
| 0.90 THz | 0 % | 3 | 7 | 1.5×10⁻⁹ |
| 0.85 THz | −5.6 % | 3 | 19 | 1.8×10⁻⁹ |
| 0.95 THz | +5.6 % | 3 | 19 | 1.2×10⁻⁹ |
| 0.80 THz | −11 % | 3 | 33 | 2.5×10⁻⁹ |
| 1.00 THz | +11 % | 3 | 30 | 1.0×10⁻⁹ |
| 1.20 THz | +33 % | 3 | 89 | 3.4×10⁻¹⁰ |
| 0.60 THz | −33 % | 3 | 139 | 3.9×10⁻⁹ |

This is across the antenna resonance (the gap field peaks near 0.8 to 0.9 THz). Each inner iteration costs one preconditioner application (two nodal solves, one FP32 triangular solve pair, two SpMVs). On the GPU that is roughly 10 to 20 ms for d289k (estimate). So reusing a factorization over a ±10 % band costs about 0.3 to 0.6 s per extra frequency, against roughly 0.5 to 1 s for a fresh factorization (Section 4.4). Refactor roughly every 10 to 20 % of frequency.

---

## 4. Recommended design, C1 (mixed-precision direct with gradient correction)

### 4.1 Components

| Component | Choice | Precision | Where |
|---|---|---|---|
| Mesh | Curved second-order tets, size field anchored to the tip, or one mesh morphed per scan position | FP64 | CPU (gmsh) |
| Assembly | $A(\omega)$ as a combination of a few frequency-independent matrices (4.3) | FP64 | CPU once per mesh, GPU combination per frequency |
| Symbolic analysis (ordering, elimination tree) | Once per mesh topology | n/a | CPU / GPU |
| Edge factorization | cuDSS, complex symmetric if supported, else general LU | FP32 | GPU |
| Gradient system | $S = G^T A G$ (nodal), factorized (cuDSS) or AMG | FP32 | GPU |
| Refinement | Stationary IR, switch to flexible GMRES-IR when more than about 10 steps or when the factor is reused at another frequency | FP64 residual and update | GPU |
| Frequency sweep | Adaptive rational model (minimal rational interpolation as in Palace, or AAA on the probe outputs) | FP64 | CPU |
| Readouts | Line integral, ball average and DFT-free frequency outputs | FP64 | GPU / CPU |

### 4.2 Algorithm per scan point

```
mesh, G            ← morph previous mesh (or remesh), discrete gradient
analysis           ← reuse if topology unchanged
choose initial frequencies f_1..f_m (e.g. 0.2, 0.5, 0.9, 1.4, 2.0 THz)
for each frequency f (adaptive):
    A  ← K + c1(f) M + c2(f) B_Au + c3(f) PML terms         (GPU axpy)
    if no factor within ±10 %:  F32 ← factor_fp32(A)        (cuDSS)
    S  ← Gᵀ A G ;  FS ← factor_fp32(S)
    x  ← GMRES-IR(A, b(f), P = grad · F32 · grad)            (FP64 residuals)
    store outputs (gap field, ball average, field on cut planes)
    update rational model, pick next frequency where its error estimate is largest
stop when the model error < tolerance on the outputs
```

### 4.3 Making assembly cheap per frequency

Today the prototype reassembles everything per frequency. In the frequency domain this is avoidable.

- **Si and air.** $k_0^2 M_\varepsilon$ is affine in $k_0^2$ (and in $\sigma/\omega$ for lossy Si).
- **Surface impedance.** $\frac{ik_0}{z_s(\omega)} B_{Au}$ is one fixed matrix times a scalar function of $\omega$.
- **PML.** The prototype scales its absorption with $1/k_0$, which makes $\Lambda$ frequency dependent in a non-affine way. Use a frequency-independent complex stretching over the sweep band instead. The attenuation then scales with $k_0$, so size the PML for the lowest frequency. That makes $K_\Lambda$ and $M_\Lambda$ fixed matrices and $A(\omega) = K_\Lambda - k_0^2 M_{\varepsilon\Lambda} - f(\omega)B_{Au}$ a combination of three stored matrices. This is a one-line change in the prototype and also makes the rational frequency model simpler.

### 4.4 Cost estimate on the RTX 2000 Ada (d289k)

| Step | Estimate | Basis |
|---|---|---|
| FP32 numeric factorization | 0.2 to 1 s | 1.6×10¹¹ MUMPS operations (up to about 6×10¹¹ real flops if counted as complex operations) at an effective 1 to 3 TFLOP/s FP32 for supernodal kernels. Mid-size matrices use the GPU poorly, hence the wide range. |
| Same factorization in FP64 on that GPU | 10 to 40 s | 1/64 throughput. Slower than the CPU. |
| Nodal factorization | under 0.1 s | Matrix 6.7× smaller |
| 3 to 8 refinement steps | 0.05 to 0.15 s | Triangular solves read the 1.3 GB factor twice per step at about 256 GB/s |
| Frequency combination of stored matrices | under 10 ms | Memory-bound axpy on 4.6M entries |
| **Per frequency** | **about 0.5 to 1.5 s** | |
| **Per scan point** (6 to 10 solves via the adaptive model, half of them recycled) | **about 5 to 15 s** | Plus meshing, see below |
| Memory | about 2 to 3 GB | 1.3 GB factor (MUMPS fill, cuDSS fill may differ, and general LU instead of $LDL^T$ doubles it), matrices, nodal factor, vectors |

**Meshing becomes the bottleneck.** gmsh needed 14 to 22 s for this mesh in the tests. Remeshing at every scan point would dominate. Mesh morphing (moving the vertices in a buffer around the tip, keeping the topology) removes this cost and also lets the symbolic analysis and sparsity pattern be reused. As argued in the landscape document, it also makes the enhancement map smooth in tip position.

**Comparison.**

| Route | Per scan point on the laptop | Status |
|---|---|---|
| CST FIT time domain | about 18 min (CPU) | Measured |
| FEniCSx frequency domain, MUMPS FP64, 20 to 30 frequencies | 7 to 10 min (CPU) | Extrapolated from measurements |
| C1 algorithm on the laptop CPU (FP32 + gradient correction + recycling + adaptive sweep) | about 0.5 to 1.5 min | Estimate from measured CPU timings (3 to 6 s per frequency, 6 to 10 solves, half of them recycled) |
| Track A, implicit FETD with AMS-CG | 1 to 2 min (GPU) | Estimate |
| **C1 on GPU** | **5 to 15 s plus meshing** | Estimate, needs the cuDSS benchmark |

### 4.5 Why this is not just "put MUMPS on the GPU"

1. The GPU is only useful in FP32 on consumer hardware.
2. FP32 alone is wrong for high-$S$ problems (Section 3.2), and residual-based checks do not detect it.
3. The gradient correction is what makes FP32 usable. Off-the-shelf mixed-precision refinement in direct solvers (cuDSS iterative refinement, MUMPS with FP32 factors) does not know about the gradient space and would not converge here, based on Section 3.2.

This combination (structure-aware mixed precision for Maxwell) is the novel and publishable core of Track C as defined here.

---

## 5. Open technical questions for C1

| Question | Why it matters | How to answer |
|---|---|---|
| Does cuDSS factor complex symmetric (non-Hermitian) matrices with $LDL^T$, or only general $LU$? | $LU$ doubles memory to about 2.6 GB FP32 (still fits) | Read the cuDSS matrix-type documentation on your machine, test both |
| cuDSS fill and speed with its own ordering on these matrices | Decides the per-frequency time | Benchmark on the exported matrices (Section 7, C-1) |
| Does pivoting in FP32 stay stable near resonance or for lossless cases? | MUMPS reported up to 49 delayed pivots in FP32 and none in FP64 | Watch the delayed pivot count and refinement steps across the sweep |
| Higher-order elements | The gradient space for N1curl order 2 is nodal P2, not P1 | Use the order-2 discrete gradient (or an auxiliary-space $\Pi$ map as in AMS), repeat 3.3 |
| Very high-Q resonances (lossless dielectric resonators) | $A$ nearly singular, FP32 factor error grows | Shift slightly into the complex plane and use as a preconditioner for the true frequency (Section 3.4 already shows this works across ±10 %) |
| Factor reuse across tip positions | Could make most scan points cost a few GMRES iterations instead of a factorization | Same test as 3.4, with morphed meshes at 0.5 to 2 µm tip shifts |
| Multiple right-hand sides | Several incidence angles or polarizations share a factorization | Block solves, almost free |

---

## 6. Scale-up variants

C1 covers the bowtie scan and similar problems up to roughly 1M DOFs on an 8 GB card (24 to 80 GB cards push this to roughly 3 to 7M DOFs, using the fill growth of Section 3.1). Beyond that, or for domains many wavelengths across (antenna arrays, metasurface patches, on-chip circuits), a different method is needed.

### 6.1 C2, two-level domain decomposition with C1 subdomain solvers

- **Method.** Overlapping Schwarz with a coarse space. Bonazzoli, Dolean, Graham, Spence and Tournier (Math. Comp. 2019) analysed two-level additive Schwarz for Maxwell with absorption, $\nabla\times\nabla\times\mathbf{E} - (k^2 + i\xi)\mathbf{E}$, using it as a preconditioner for the problem without absorption. GMRES converges in a $k$-independent number of iterations for $|\xi| \sim k^2$ with subdomain and coarse sizes of order $1/k$. For small or zero absorption they report alternative, more robust preconditioners as heuristics. The coarse space can use lower-order elements than the fine space.
- **Transmission conditions.** Optimized Schwarz with impedance or higher-order conditions (Dolean, Gander and Gerardo-Giorda 2009) improves convergence for propagative problems.
- **Sweeping.** The double sweep of Vion and Geuzaine (ESAIM Proc. 2018) gives nearly $k$-independent convergence for Helmholtz and Maxwell on layered decompositions when the interface DtN maps are approximated well (PML or high-order conditions). The sweeps are sequential, which limits parallelism, but pipelining many right-hand sides recovers it (they estimate about 25× for many sources).
- **GPU mapping.** Subdomain problems are independent mid-size factorizations, which suit batched or multi-stream cuDSS in FP32, each with its own gradient correction. The coarse problem stays in FP64.
- **Evidence for the GPU approach.** Rotem (2026) ran a non-overlapping DD for Helmholtz on an A100 and an RTX 4060 Ti, with each subdomain on one thread block, WaveHoltz as the subdomain solver and the whole cycle in a single kernel. FP32 gave an extra 2 to 10× over FP64 (5 to 10× on the RTX card). The study is limited to scalar Helmholtz, without a coarse space, with small subdomains and a single GPU.
- **Open problems.** A robust coarse space for propagative Maxwell with PML, and the interaction between the gradient near-null space and the subdomain boundaries (the coarse space must also contain gradients, as in GenEO-type constructions for H(curl)).

### 6.2 C3, matrix-free iterative solver with shifted auxiliary-space multigrid

- **Method.** FGMRES on $A(\omega)$, preconditioned by an auxiliary-space multigrid (AMS-type) applied to a shifted, damped operator $K - (k_0^2 + i\xi)M$, or p-multigrid with partial assembly and a complex coarse solve. Palace already offers the ingredients (`PCMatShifted`, `PCMatReal`, `ComplexCoarseSolve`, geometric p/h-multigrid, GPU partial assembly).
- **Strength.** Memory is linear in DOFs and all kernels are memory-bound, so FP64 is fine on consumer GPUs.
- **Weakness.** Iteration counts grow with $kL$ and with the PML. For the bowtie ($kL$ small) it may be competitive, which is worth one test with Palace or with PETSc and hypre (FEniCSx real build, using the real-equivalent 2×2 block form, since hypre does not support complex PETSc).
- **Hybrid.** Use C1 on a coarse level (FP32 factorization of the order-1 problem) inside p-multigrid for higher-order elements. This combines C1's robustness with matrix-free fine levels.

#### 6.2.1 State of the art at extreme scale (Fu 2026)

Fu (arXiv 2608.22903, August 2026, preprint) is the strongest current demonstration of the C3 approach.

- **Method.** Lowest-order Nédélec elements on nested Cartesian hexahedral grids (h, 2h, 4h) with a Cartesian PML and optimally blended quadrature (fourth-order dispersion error at no extra cost). The outer FGMRES(4) works on the unshifted operator. The preconditioner smooths on the fine grid with GMRES(4)-Jacobi(2), then corrects with an inner FGMRES on the unshifted 2h operator. That inner FGMRES is preconditioned by a V-cycle on shifted operators $C - (1+i\sigma)k_0^2 M$ with $\sigma = 0.4$ on the 2h and 4h grids. The 4h bottom level is also iterative. There is no factorization and no stored matrix, and the parameters come from a local Fourier analysis.
- **Results.** Complex single precision throughout, outer tolerance 10⁻³, 8 points per wavelength, domains of about 93 to 370 wavelengths. About 1.7×10⁸ unknowns took 9.6 to 13 s on one A100-40G (37 GB), and 1.09×10¹⁰ unknowns took 42 to 72 s on 64 A100 GPUs. That needed 18 to 98 preconditioner applications, with 70 to 91 % weak-scaling efficiency. Eight A100s were 41× faster than 768 CPU ranks and used 7.1× less memory. Single and double precision needed the same 30 preconditioner applications and differed by 3.8×10⁻⁵.
- **Shared with this document.** The shift acts only inside the preconditioner while the outer iteration and the stopping test use the true operator. That is the same principle as frequency recycling in Section 3.4. The heavy kernels run in FP32, and a flexible outer Krylov method absorbs the nonlinearity of fixed-work inner iterations, as in the GMRES-IR loop used here.

**Why it does not transfer directly to the bowtie.**

| Aspect | Fu 2026 | Bowtie near-field problem |
|---|---|---|
| Resolution | 8 points per wavelength, $k_0 h \approx 0.8$ | 10⁴ to 10⁵ points per wavelength near the tip, $k_0 h \approx 10^{-4}$ |
| Domain size | 93 to 370 wavelengths | 1.5 to 5 wavelengths |
| Dominant ill-conditioning | Wave indefiniteness | Low-frequency breakdown (gradient near-null space) |
| Gradient to curl eigenvalue ratio, $(k_0 h)^2$ | about 0.6, handled by the shift and Jacobi smoothing | about 10⁻⁸, Jacobi cannot reduce gradient errors |
| Mesh | Uniform structured hexahedra, geometric coarsening | Graded curved tetrahedra, no nested hierarchy |
| Materials | Smooth, impedance-matched ($\varepsilon_r = \mu_r$), no metals | PEC tip, Au surface impedance, Si (ε = 11.7) against air |
| Precision check | FP32 validated at residual 10⁻³ | FP32 alone gave 16 % to 8,500 % error at residuals of 10⁻³ to 10⁻⁶ (Section 3.2) |
| Hardware | 40 GB A100 and up | 8 GB laptop GPU |

The paper's single-precision result holds in its own well-conditioned regime. In the high-$S$ regime a small residual does not guarantee a small error, so the residual must be computed in FP64 and the gradient space must be treated explicitly.

**What to take from it.**
- For the bowtie, C1 remains the better choice. A 0.3M DOF problem a few wavelengths across is cheap to factorize, and the paper's advantages appear only at very large size.
- For large problems (arrays, long tips, metasurface patches, many wavelengths), the paper's design is the starting point, with two changes for multi-scale geometry. Replace Jacobi with a Hiptmair-type hybrid smoother (edge smoothing plus nodal smoothing on the gradient space) so that strongly refined regions are handled. Replace geometric coarsening with auxiliary-space or algebraic coarsening that works on unstructured, graded tetrahedra.
- **Hybrid domain decomposition (an extension of C2).** Solve a structured hexahedral background region matrix-free in the paper's style. Couple it through Schwarz domain decomposition to an unstructured tetrahedral near zone around the tip and gap, solved with C1 (FP32 direct plus gradient correction). Each solver then works in its own regime. I am not aware of a published combination of this kind.

### 6.3 C4, EM-WaveHoltz on the Track A time stepper

- **Method.** Peng and Appelö (IEEE TAP 2022) compute time-harmonic Maxwell solutions by running the time-domain solver over a few periods, filtering in time, and iterating. The resulting operator is positive definite (CG or GMRES), with convergence guaranteed away from resonances. For the scalar WaveHoltz method with implicit time stepping, Henshaw reports that about 10 steps per period suffice.
- **GPU fit.** It reuses the Track A kernel (implicit FETD with AMS-CG), so every operation is memory-bound and FP64 is affordable. A 2026 dissertation (Rotem) reports a GPU implementation with multigrid-accelerated time stepping and a GPU-tailored DD with WaveHoltz subdomain solves (no benchmarks in the abstract).
- **Cost estimate for the bowtie.** 20 to 50 Krylov iterations × 2 periods × 10 steps × 5 to 15 CG iterations × about 5 ms gives about 20 to 75 s per frequency on the laptop GPU. That is slower than C1 here but scales to problems C1 cannot hold.
- **Risks.** Slow convergence near lightly damped resonances. The bowtie resonance is radiatively damped (low Q), but the Si slab supports guided modes that run into the PML only laterally. Test across 0.7 to 1.0 THz.

### 6.4 Model order reduction for sweeps (applies to all variants)

- Palace's adaptive driven solver builds a projection-based reduced model from a few adaptively chosen full solves and evaluates it on a dense output grid. Its documentation mentions minimal rational interpolation and recommends validating against a uniform sweep for each new model.
- AAA rational approximation of the scalar outputs (gap field, ball average) is a simple alternative that needs no access to the solver internals.
- The bowtie response is smooth except near the resonance and the slab Fabry-Pérot features, so 6 to 10 samples should suffice between 0.2 and 2 THz. This needs checking against the uniform sweep from the prototype.

---

## 7. Development plan

| Phase | Content | Exit criterion |
|---|---|---|
| **C-0 (done)** | Matrix export tooling, CPU experiments with FP32 MUMPS, gradient correction, frequency recycling | Sections 3.1 to 3.4 |
| **C-1. cuDSS benchmark (1 to 2 weeks)** | Install the CUDA toolkit and cuDSS in WSL2. Run cuDSS on the exported matrices (c65k, f138k, d289k) in FP32 and FP64, $LDL^T$ and $LU$. Record fill, memory, analysis and factorization time, delayed pivots. Python access through NVIDIA's nvmath-python (check its sparse direct solver interface) or CUDSS.jl, or a small C program. | FP32 factorization of d289k under 2 s and under 4 GB. If it takes over 5 s, C1 loses its edge over Track A. |
| **C-2. GPU refinement loop (2 weeks)** | Gradient correction and GMRES-IR on the GPU (CuPy or nvmath for SpMV and vectors) | Same step counts and errors as Section 3.3 |
| **C-3. Integration (3 to 4 weeks)** | Affine frequency assembly, frequency-independent PML, recycling policy, adaptive rational sweep, readouts | Broadband enhancement spectrum matching the uniform FP64 sweep within 0.5 % |
| **C-4. Scan workflow (3 to 4 weeks)** | Mesh morphing, analysis reuse, factor reuse across tip positions (test 5 in Section 5) | Smooth map, measured time per scan point |
| **C-5. Decision against Track A** | Same scan points with both routes | Pick the faster for production, keep the other as cross-validation |
| **C-6. Scale-up (research, months)** | C2 or C3 or C4 on a larger problem (bowtie array, longer tip, multi-λ domain) | Publication-quality comparison |

**Integration targets.**
- *FEniCSx.* The prototype already produces the matrices, and `dolfinx.fem.petsc.discrete_gradient` gives $G$. cuDSS is called outside PETSc through Python bindings.
- *MFEM / Palace.* MFEM provides a cuDSS solver interface (`mfem::CuDSSSolver`), and Palace runs on MFEM with complex frequency-domain systems and AMS. The gradient-corrected FP32 refinement would be a new preconditioner option there, which fits the plan in the landscape document to contribute to Palace.

---

## 8. Risks

1. **cuDSS performance on mid-size matrices.** Sparse direct solvers use GPUs well only for large fronts. At 289k DOFs many fronts are small, so the effective FLOP rate may be far below peak. Phase C-1 measures this directly.
2. **Complex symmetric support.** If cuDSS only offers general $LU$ for complex non-Hermitian matrices, memory and time roughly double. It still fits for this problem.
3. **FP32 pivoting near resonance.** Delayed pivots appeared in FP32 and not in FP64. Static pivoting with refinement is a fallback, and the gradient correction should absorb the resulting perturbation, but this needs testing over the full sweep.
4. **Generality of the gradient correction.** It relies on the exact discrete de Rham structure (N1curl with a matching nodal space). Non-conforming couplings, mortar interfaces or DG formulations need their own versions.
5. **Estimates.** All GPU timings in this document are estimates. The memory and convergence results are measured, but on CPU MUMPS.

---

## 9. How Track C fits with Track A

| Situation | Better route |
|---|---|
| Broadband scans, PEC or simple metals, moderate size (this bowtie) | C1 if the cuDSS benchmark confirms under about 2 s per factorization, otherwise Track A |
| Dispersive materials, measured $\varepsilon(\omega)$, 2D sheets, exact surface impedance | Track C |
| Single or few frequencies, adjoint optimization, eigenmodes | Track C |
| Very large domains (many wavelengths, arrays, more than 2M DOFs on a laptop) | Track A, or C2/C4 as research |
| Time traces for comparison with THz-STM or EOS waveforms | Track A, or Track C with an inverse FFT of a dense sweep |

Both tracks share the mesh, the geometry, the gradient operator, the readouts and the validation suite. The cross-check between them (time-domain DFT against frequency-domain solves on the same mesh) is itself a strong verification test.

---

## 10. Reproducing the experiments

The scripts are in `fenicsx/trackc/` of the project.

| Script | Purpose |
|---|---|
| `dump_mats.py` | Assembles the bowtie system with the prototype at chosen frequencies, saves $A$, $b$, the FP64 solution and the discrete gradient $G$ as `.npz` (directory set by `MATS`, default `mats/`) |
| `run_ir.py` | FP64 versus FP32 MUMPS factorization, memory and time, plain refinement and GMRES-IR |
| `run_grad.py` | FP32 factorization with gradient correction, stationary refinement (set `LIGHT=1` for low-memory mode) |
| `run_recycle.py` | Reuses the factorization at one frequency for others, GMRES-IR with right preconditioning |

Environment. `dump_mats.py` runs in the FEniCSx environment (complex PETSc, dolfinx 0.11). The other scripts need a separate environment with `python-mumps`, SciPy and NumPy (conda-forge), because they factor in complex single precision, which the complex-double PETSc build cannot do. On the laptop, the same `.npz` files are the input for the cuDSS benchmark of phase C-1.

Example.
```
python dump_mats.py d289k 0.9 -- --out res                 # production mesh
python dump_mats.py c65k 0.3 0.9 2.0 -- --h-tip 0.03 --h-gap 0.03 --h-edge 0.5 --growth 0.4 --h-air 25 --h-si 15
python run_ir.py mats/d289k_0.90.npz
python run_grad.py mats/c65k_G.npz mats/c65k_0.30.npz mats/c65k_0.90.npz
python run_recycle.py mats/c65k_G.npz mats/c65k_0.90.npz mats/c65k_0.85.npz mats/c65k_1.00.npz
```

---

## References

**Domain decomposition and preconditioning**
- M. Bonazzoli, V. Dolean, I. G. Graham, E. A. Spence, P.-H. Tournier, "Domain decomposition preconditioning for the high-frequency time-harmonic Maxwell equations with absorption", Math. Comp. 88 (2019). https://arxiv.org/abs/1711.03789
- V. Dolean, M. J. Gander, L. Gerardo-Giorda, "Optimized Schwarz methods for Maxwell's equations", SIAM J. Sci. Comput., 2009.
- A. Vion, C. Geuzaine, "Improved sweeping preconditioners for domain decomposition algorithms applied to time-harmonic Helmholtz and Maxwell problems", ESAIM: Proc. Surveys 61 (2018) 93-111. https://www.esaim-proc.org/articles/proc/pdf/2018/01/proc_esaim2018_093.pdf
- A. Rotem, "Efficient domain decomposition for the Helmholtz equation on GPUs", arXiv:2606.21695 (2026). https://arxiv.org/abs/2606.21695
- R. Hiptmair, J. Xu, "Nodal auxiliary space preconditioning in H(curl) and H(div) spaces", SIAM J. Numer. Anal., 2007.
- R. Hiptmair, "Multigrid method for Maxwell's equations", SIAM J. Numer. Anal., 1998.
- Tz. Kolev, P. Vassilevski, "Parallel auxiliary space AMG for H(curl) problems", J. Comput. Math., 2009.

**Mixed precision and direct solvers**
- P. Amestoy, A. Buttari, N. J. Higham, J.-Y. L'Excellent, T. Mary, B. Vieublé, "Combining sparse approximate factorizations with mixed precision iterative refinement", ACM Trans. Math. Softw., 2023. Preprint https://eprints.maths.manchester.ac.uk/2845/
- E. Carson, N. J. Higham, "Accelerating the solution of linear systems by iterative refinement in three precisions", SIAM J. Sci. Comput., 2018.
- NVIDIA cuDSS documentation (data types, downcast factors, iterative refinement, hybrid memory). https://docs.nvidia.com/cuda/cudss/latest/types.html
- STRUMPACK GPU support (direct solver only, BLR and HSS on CPU). https://portal.nersc.gov/project/sparse/strumpack/master/GPU_Support.html
- MFEM `CuDSSSolver`. https://docs.mfem.org/4.10/classmfem_1_1CuDSSSolver.html

**Matrix-free multigrid at extreme scale**
- S. Fu, "A fully matrix-free three-grid preconditioner for the time-harmonic Maxwell equations at extreme scale", arXiv:2608.22903 (2026), preprint. https://arxiv.org/abs/2608.22903

**Time-domain based frequency solvers**
- Z. Peng, D. Appelö, "EM-WaveHoltz: A flexible frequency-domain method built from time-domain solvers", IEEE Trans. Antennas Propag., 2022. https://arxiv.org/abs/2103.14789
- A. Rotem, "WaveHoltz: Theory and High Performance Applications", PhD dissertation, Virginia Tech, 2026. https://vtechworks.lib.vt.edu/items/184a4e65-ba14-48ba-be8b-2f0a4c5decc5
- W. Henshaw, WaveHoltz and EigenWave seminar slides (implicit time stepping with about 10 steps per period, deflation). https://mfem.org/pdf/seminar/henshaw.pdf

**Frequency sweeps**
- Palace, adaptive frequency sweeps for driven simulations. https://awslabs.github.io/palace/stable/features/adaptive_driven_solver/
- Palace solver configuration (shifted and real preconditioner options, complex coarse solve, multigrid). https://awslabs.github.io/palace/dev/config/solver
- Y. Nakatsukasa, O. Sète, L. N. Trefethen, "The AAA algorithm for rational approximation", SIAM J. Sci. Comput., 2018.
