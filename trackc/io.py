"""npz import/export of a system (matrices stored as CSR components under a prefix)."""
import numpy as np
import scipy.sparse as sp
from .system import AffineSystem


def _put(d, name, M):
    M = sp.csr_matrix(M)
    d[f"{name}_data"], d[f"{name}_indices"], d[f"{name}_indptr"] = M.data, M.indices, M.indptr
    d[f"{name}_shape"] = np.array(M.shape)


def _get(z, name):
    return sp.csr_matrix((z[f"{name}_data"], z[f"{name}_indices"], z[f"{name}_indptr"]), shape=tuple(z[f"{name}_shape"]))


def save_fixed(path, A, G, b):
    d = {"b": b}
    _put(d, "A", A)
    _put(d, "G", G)
    np.savez_compressed(path, **d)


def load_fixed(path):
    """Returns (A, G, b). Expects the layout written by save_fixed."""
    z = np.load(path)
    return _get(z, "A"), _get(z, "G"), z["b"]


def save_affine(path, terms, G, b):
    d = {"b": b, "nterms": np.array(len(terms))}
    for i, T in enumerate(terms):
        _put(d, f"T{i}", T)
    _put(d, "G", G)
    np.savez_compressed(path, **d)


def load_affine_terms(path):
    z = np.load(path)
    return [_get(z, f"T{i}") for i in range(int(z["nterms"]))], _get(z, "G"), z["b"]
