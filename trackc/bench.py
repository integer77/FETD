"""Benchmark on the synthetic problem: python -m trackc.bench [n_half] [power] [lambda]"""
import sys
import time
import numpy as np
from . import problem, solver
from .factor import Factorization
from .sweep import FrequencySweep


def main(n_half=10, power=4.0, lam=1.0):
    p = problem.build(n_half=n_half, power=power)
    k = 2 * np.pi / lam
    A = p.A(k)
    print(f"DOFs {p.n_dof}, nodal DOFs {p.G.shape[1]}, lambda/h_min {lam / p.h_min:.3g}")
    t = time.time()
    F64 = Factorization(A, np.complex128)
    print(f"FP64 factor {time.time() - t:.1f} s, {F64.nbytes / 2**20:.0f} MB, backend {F64.backend}")
    xref = F64.solve(p.b)
    t = time.time()
    F = Factorization(A)
    print(f"FP32 factor {time.time() - t:.1f} s, {F.nbytes / 2**20:.0f} MB, delayed pivots {F.delayed_pivots}")
    err = lambda x: np.linalg.norm(x - xref) / np.linalg.norm(xref)
    x = F.solve(p.b)
    print(f"plain FP32: residual {np.linalg.norm(p.b - A @ x) / np.linalg.norm(p.b):.1e}, error {err(x):.1e}")
    r = solver.refine(A, p.b, lambda v: F.solve(v), maxit=10)
    print(f"classical IR: converged={r.converged}, final residual {r.resid_history[-1]:.1e}")
    for name, nd in (("FP64", np.complex128), ("FP32", np.complex64)):
        P = solver.GradientCorrectedPrecond(A, p.G, edge_factor=F, nodal_dtype=nd)
        r = solver.refine(A, p.b, P)
        print(f"gradient-corrected IR (nodal {name}): converged={r.converged}, steps {r.steps}, error {err(r.x):.1e}")
    sw = FrequencySweep(p, reuse_tol=0.05)
    for f in (1.0, 1.02, 1.04, 1.06, 1.3):
        pt = sw.solve(k * f)
        print(f"sweep k x{f}: refactored={pt.refactored}, steps {pt.steps}, inner iters {pt.inner_iters}, converged={pt.converged}")


if __name__ == "__main__":
    main(*(int(sys.argv[1]) if len(sys.argv) > 1 else 10, float(sys.argv[2]) if len(sys.argv) > 2 else 4.0,
           float(sys.argv[3]) if len(sys.argv) > 3 else 1.0))
