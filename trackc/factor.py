"""Complex symmetric LDL^T in a chosen precision behind a small interface.

Backends: sequential MUMPS through a tiny C shim (stand-in for cuDSS, same fill-reducing
behaviour class), with scipy SuperLU as a slow fallback. cuDSS would implement the same
Factorization interface (constructor from a sparse matrix, .solve(r), .nbytes).
"""
import ctypes
import os
import subprocess
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = None


def _load_shim():
    global _LIB
    if _LIB is not None:
        return _LIB or None
    so = os.path.join(_HERE, "libmumps_shim.so")
    try:
        if not os.path.exists(so):
            subprocess.run(
                ["gcc", "-O2", "-shared", "-fPIC", "-I/usr/include/mumps_seq", os.path.join(_HERE, "mumps_shim.c"),
                 "-o", so, "-lcmumps_seq", "-lzmumps_seq", "-lmpiseq_seq"],
                check=True, capture_output=True)
        lib = ctypes.CDLL(so)
        lib.mf_create.restype = ctypes.c_void_p
        lib.mf_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_long] + [ctypes.c_void_p] * 3 + [ctypes.c_void_p]
        lib.mf_solve.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.mf_free.argtypes = [ctypes.c_void_p]
        _LIB = lib
    except (OSError, subprocess.CalledProcessError, FileNotFoundError):
        _LIB = False
    return _LIB or None


class Factorization:
    """Diagonally scaled LDL^T (complex symmetric) of a sparse matrix in `dtype`.

    solve() takes and returns complex128; the cast to `dtype` happens inside.
    """

    def __init__(self, A, dtype=np.complex64, backend="auto"):
        self.dtype = np.dtype(dtype)
        A = sp.csc_matrix(A)
        d = np.abs(A.diagonal())
        d[d == 0] = 1.0
        self.scale = 1.0 / np.sqrt(d)
        S = sp.diags(self.scale)
        As = (S @ A @ S).tocsr()
        self.n = A.shape[0]
        lib = _load_shim() if backend in ("auto", "mumps") else None
        if lib is not None:
            self._init_mumps(lib, As)
        elif backend == "mumps":
            raise RuntimeError("MUMPS shim unavailable")
        else:
            self.lib = None
            self.lu = spla.splu(As.astype(self.dtype).tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=1.0)
            self.nnz_lu = self.lu.L.nnz + self.lu.U.nnz
            self.nbytes = self.nnz_lu * (self.dtype.itemsize + 4)
        self.backend = "mumps" if self.lib else "superlu"

    def _init_mumps(self, lib, As):
        U = sp.triu(As).tocoo()
        self.lib = lib
        self._irn = (U.row + 1).astype(np.int32)
        self._jcn = (U.col + 1).astype(np.int32)
        self._a = U.data.astype(self.dtype)
        info = np.zeros(4, np.int32)
        prec = 0 if self.dtype == np.complex64 else 1
        self._h = lib.mf_create(prec, self.n, len(self._a), self._irn.ctypes.data, self._jcn.ctypes.data,
                                self._a.ctypes.data, info.ctypes.data)
        if info[0] < 0:
            raise RuntimeError(f"MUMPS factorization failed, INFOG(1)={info[0]}")
        self.nnz_lu = int(info[1])                      # entries of the factor (INFOG(29))
        self.nbytes = int(info[2]) * 2**20             # estimated working memory in MB (INFOG(19))
        self.delayed_pivots = int(info[3])             # INFOG(13)
        self._buf = None

    def __del__(self):
        if getattr(self, "lib", None) and getattr(self, "_h", None):
            self.lib.mf_free(self._h)
            self._h = None

    def solve(self, r):
        rs = (self.scale * r).astype(self.dtype)
        if self.lib:
            rs = np.ascontiguousarray(rs)
            if self.lib.mf_solve(self._h, rs.ctypes.data) < 0:
                raise RuntimeError("MUMPS solve failed")
            y = rs
        else:
            y = self.lu.solve(rs)
        return self.scale * y.astype(np.complex128)
