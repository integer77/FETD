"""Robustness scan: refinement steps and error vs size, scale separation and wavelength.

python -m trackc.scaling
"""
import time
import numpy as np
from . import problem, solver
from .factor import Factorization


def run(n_half, power, lam):
    p = problem.build(n_half=n_half, power=power)
    A = p.A(2 * np.pi / lam)
    t = time.time()
    F64 = Factorization(A, np.complex128)
    t64 = time.time() - t
    xref = F64.solve(p.b)
    del F64
    t = time.time()
    F = Factorization(A)
    t32 = time.time() - t
    err = lambda x: np.linalg.norm(x - xref) / np.linalg.norm(xref)
    plain = err(F.solve(p.b))
    P = solver.GradientCorrectedPrecond(A, p.G, edge_factor=F)
    r = solver.refine(A, p.b, P)
    print(f"n={p.n_dof:7d} S={lam / p.h_min:9.3g} lam={lam:5.2f} plain_err={plain:8.1e} "
          f"steps={r.steps:2d} conv={r.converged} err={err(r.x):8.1e} "
          f"t64={t64:5.1f}s t32={t32:5.1f}s MB32={F.nbytes >> 20}", flush=True)


if __name__ == "__main__":
    for nh in (8, 10, 12):                      # size
        run(nh, 4.0, 1.0)
    for pw in (3.0, 4.0, 5.0):                  # scale separation at fixed size
        run(10, pw, 1.0)
    for lam in (3.0, 1.0, 0.5, 0.3):            # wavelength at fixed mesh
        run(10, 4.0, lam)
