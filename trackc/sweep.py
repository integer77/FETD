"""Frequency sweep with FP32 factor recycling across nearby wavenumbers."""
from dataclasses import dataclass
import numpy as np
from .factor import Factorization
from .solver import GradientCorrectedPrecond, refine, gmres_ir


@dataclass
class SweepPoint:
    k: float
    x: np.ndarray
    refactored: bool
    steps: int
    inner_iters: int
    converged: bool


class FrequencySweep:
    """Solves A(k) x = b for a list of k, refactoring only when k leaves reuse_tol of the last factor."""

    def __init__(self, problem, reuse_tol=0.12, tol=1e-10):
        self.p, self.reuse_tol, self.tol = problem, reuse_tol, tol
        self.k_ref = None
        self.F = None
        self.factorizations = 0

    def solve(self, k):
        A = self.p.A(k)
        refactored = self.k_ref is None or abs(k - self.k_ref) > self.reuse_tol * self.k_ref
        if refactored:
            self.F = Factorization(A)
            self.k_ref = k
            self.factorizations += 1
        P = GradientCorrectedPrecond(A, self.p.G, edge_factor=self.F)
        if refactored:
            res = refine(A, self.p.b, P, self.tol)
        else:
            res = gmres_ir(A, self.p.b, P, self.tol)
        return SweepPoint(k, res.x, refactored, res.steps, res.inner_iters, res.converged)

    def run(self, ks):
        return [self.solve(k) for k in ks]
