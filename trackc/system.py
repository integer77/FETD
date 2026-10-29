"""Problem-independent inputs: an affine-in-frequency system A(k) = sum_i c_i(k) T_i, plus the discrete gradient G.

Any H(curl) discretization can be wrapped: provide the frequency-independent matrices T_i (curl-curl, masses,
boundary terms, ...), their scalar coefficient functions, and G (edges x nodes, with the nodal space chosen so
that (curl-curl) G = 0 on the free edges, e.g. phi = 0 on PEC conductors).
"""
from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np
import scipy.sparse as sp


@dataclass
class AffineSystem:
    terms: Sequence[sp.spmatrix]
    coeffs: Sequence[Callable[[float], complex]]
    G: sp.spmatrix
    b: np.ndarray                     # (n,) or (n, nrhs)

    def A(self, k):
        out = self.coeffs[0](k) * sp.csr_matrix(self.terms[0], dtype=complex)
        for T, c in zip(self.terms[1:], self.coeffs[1:]):
            out = out + c(k) * sp.csr_matrix(T, dtype=complex)
        return out.tocsr()


def maxwell_system(K, M, B, G, b):
    """Standard form A(k) = K - k^2 M - i k B (k = omega/c)."""
    return AffineSystem([K, M, B], [lambda k: 1.0, lambda k: -k * k, lambda k: -1j * k], G, b)


def fixed_matrix_system(A, G, b):
    """Single frequency, already assembled A."""
    return AffineSystem([A], [lambda k: 1.0], G, b)
