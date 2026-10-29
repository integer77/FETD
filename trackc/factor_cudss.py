"""cuDSS backend through nvmath-python. UNTESTED: written against the nvmath 1.0 API without a GPU available.

Same interface as trackc.factor.Factorization, so it plugs in as
MixedPrecisionSolver(system, factorization_cls=CuDSSFactorization).
Operands are passed as host SciPy/NumPy; nvmath moves them. Moving the RHS to device memory once and keeping
vectors on the GPU (CuPy) is the next optimisation, together with a device-side refinement loop.
"""
import numpy as np
import scipy.sparse as sp


class CuDSSFactorization:
    def __init__(self, A, dtype=np.complex64, symmetric=True):
        import nvmath.sparse.advanced as nvs
        self.dtype = np.dtype(dtype)
        A = sp.csc_matrix(A)
        d = np.abs(A.diagonal())
        d[d == 0] = 1.0
        self.scale = 1.0 / np.sqrt(d)
        S = sp.diags(self.scale)
        As = (S @ A @ S).tocsr().astype(self.dtype)
        self.n = A.shape[0]
        # complex symmetric (not Hermitian) is the open question from the design doc; fall back to general LU
        if symmetric:
            opts = nvs.DirectSolverOptions(sparse_system_type=nvs.DirectSolverMatrixType.SYMMETRIC)
            a_used = sp.triu(As).tocsr()
            opts.sparse_system_view = nvs.DirectSolverMatrixViewType.UPPER
        else:
            opts = nvs.DirectSolverOptions()
            a_used = As
        self._solver = nvs.DirectSolver(a_used, np.zeros(self.n, self.dtype), options=opts, execution="cuda")
        self._solver.plan()
        self._solver.factorize()
        self.backend = "cudss"
        self.nnz_lu = 0
        self.nbytes = 0
        self.delayed_pivots = 0

    def solve(self, r):
        rs = (self.scale * r).astype(self.dtype)
        self._solver.reset_operands(b=rs)
        y = np.asarray(self._solver.solve())
        return self.scale * y.astype(np.complex128)

    def __del__(self):
        s = getattr(self, "_solver", None)
        if s is not None:
            s.free()
