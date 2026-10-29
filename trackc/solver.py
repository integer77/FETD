"""Gradient-corrected FP32 preconditioner with FP64 iterative refinement and flexible GMRES-IR."""
from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp
from .factor import Factorization


@dataclass
class SolveResult:
    x: np.ndarray
    converged: bool
    steps: int
    inner_iters: int
    resid_history: list = field(default_factory=list)


class GradientCorrectedPrecond:
    """P(r): y = G S^-1 G^T r; y += F^-1 (r - A y); y += G S^-1 G^T (r - A y).

    F is an FP32 factor of the edge matrix (possibly built at another frequency);
    S = G^T A G is the nodal system, rebuilt cheaply for the current A.
    """

    def __init__(self, A, G, edge_factor=None, dtype=np.complex64, nodal_dtype=np.complex64):
        self.G = sp.csr_matrix(G)
        self.GT = self.G.T.tocsr()
        self.F = edge_factor if edge_factor is not None else Factorization(A, dtype)
        self.nodal_dtype = nodal_dtype
        self.rebind(A)

    def rebind(self, A):
        """Point the preconditioner at a new matrix (e.g. another frequency), reusing the edge factor."""
        self.A = sp.csr_matrix(A)
        S = (self.GT @ self.A @ self.G).tocsr()
        self.S = Factorization(S, self.nodal_dtype)

    def _grad(self, r):
        return self.G @ self.S.solve(self.GT @ r)

    def __call__(self, r):
        y = self._grad(r)
        y = y + self.F.solve(r - self.A @ y)
        y = y + self._grad(r - self.A @ y)
        return y


def refine(A, b, P, tol=1e-10, maxit=30):
    """Stationary iterative refinement, residual and update in FP64."""
    bn = np.linalg.norm(b)
    x = np.zeros_like(b)
    hist = []
    for it in range(maxit):
        r = b - A @ x
        rel = np.linalg.norm(r) / bn
        hist.append(rel)
        if rel < tol:
            return SolveResult(x, True, it, it, hist)
        x = x + P(r)
    r = b - A @ x
    hist.append(np.linalg.norm(r) / bn)
    return SolveResult(x, hist[-1] < tol, maxit, maxit, hist)


def _fgmres(A, r, P, tol, m):
    """Flexible GMRES(m), right preconditioned, from a zero initial guess. Returns (dx, iters)."""
    n = len(r)
    beta = np.linalg.norm(r)
    V = np.zeros((m + 1, n), complex)
    Z = np.zeros((m, n), complex)
    H = np.zeros((m + 1, m), complex)
    V[0] = r / beta
    e1 = np.zeros(m + 1, complex)
    e1[0] = beta
    y, k = None, 0
    for j in range(m):
        Z[j] = P(V[j])
        w = A @ Z[j]
        for _ in range(2):                                   # modified Gram-Schmidt with reorthogonalization
            for i in range(j + 1):
                h = np.vdot(V[i], w)
                H[i, j] += h
                w = w - h * V[i]
        H[j + 1, j] = np.linalg.norm(w)
        k = j + 1
        y = np.linalg.lstsq(H[:k + 1, :k], e1[:k + 1], rcond=None)[0]
        if np.linalg.norm(e1[:k + 1] - H[:k + 1, :k] @ y) / beta < tol or H[j + 1, j] == 0:
            break
        V[j + 1] = w / H[j + 1, j]
    return Z[:k].T @ y, k


def gmres_ir(A, b, P, tol=1e-10, inner_tol=1e-6, restart=60, max_outer=10):
    """Outer refinement on the true FP64 residual, inner right-preconditioned FGMRES."""
    bn = np.linalg.norm(b)
    x = np.zeros_like(b)
    hist, total = [], 0
    for it in range(max_outer):
        r = b - A @ x
        rel = np.linalg.norm(r) / bn
        hist.append(rel)
        if rel < tol:
            return SolveResult(x, True, it, total, hist)
        dx, k = _fgmres(A, r, P, inner_tol, restart)
        x = x + dx
        total += k
    hist.append(np.linalg.norm(b - A @ x) / bn)
    return SolveResult(x, hist[-1] < tol, max_outer, total, hist)
