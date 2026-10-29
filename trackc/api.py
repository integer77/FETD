"""High-level entry point: solve A(k) x = b for any system with .A(k), .G and .b."""
import numpy as np
from .factor import Factorization
from .solver import GradientCorrectedPrecond, refine, gmres_ir, SolveResult


class MixedPrecisionSolver:
    """Reuses one FP32 factorization across nearby k (gmres_ir) and refactors when k drifts by more than reuse_tol.

    `factorization_cls` is the backend; swap in a GPU class with the same interface.
    """

    def __init__(self, system, reuse_tol=0.05, tol=1e-10, factorization_cls=Factorization):
        self.sys, self.reuse_tol, self.tol, self.cls = system, reuse_tol, tol, factorization_cls
        self.k_ref = self.F = self.P = None
        self.factorizations = 0

    def solve(self, k, b=None):
        """Returns SolveResult (x has shape of b; multiple right-hand sides share factors)."""
        b = self.sys.b if b is None else b
        A = self.sys.A(k)
        fresh = self.k_ref is None or abs(k - self.k_ref) > self.reuse_tol * self.k_ref
        if fresh:
            self.F = self.cls(A)
            self.k_ref = k
            self.factorizations += 1
            self.P = GradientCorrectedPrecond(A, self.sys.G, edge_factor=self.F)
        else:
            self.P.rebind(A)
        run = refine if fresh else gmres_ir
        cols = [b] if b.ndim == 1 else list(b.T)
        res = [run(A, c, self.P, self.tol) for c in cols]
        x = res[0].x if b.ndim == 1 else np.stack([r.x for r in res], axis=1)
        return SolveResult(x, all(r.converged for r in res), max(r.steps for r in res),
                           sum(r.inner_iters for r in res), res[0].resid_history)
