"""Lowest-order Nedelec (Whitney) edge elements on tetrahedra, vectorized assembly.

Basis of edge (i<j in global vertex numbering): w = l_i grad l_j - l_j grad l_i.
The DOF is the tangential line integral, so the discrete gradient G has entries -1/+1.
"""
import numpy as np
import scipy.sparse as sp

_LOC_EDGES = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
_LOC_FACES = np.array([(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)])


class EdgeTable:
    def __init__(self, tets, nv):
        self.nv = nv
        pairs = np.sort(tets[:, _LOC_EDGES], axis=2)           # (nt,6,2) sorted global pairs
        keys = pairs[..., 0].astype(np.int64) * nv + pairs[..., 1]
        self.keys, inv = np.unique(keys.ravel(), return_inverse=True)
        self.tet_edges = inv.reshape(keys.shape)               # (nt,6)
        self.edges = np.stack([self.keys // nv, self.keys % nv], axis=1)
        self.n = len(self.keys)

    def lookup(self, i, j):
        k = np.minimum(i, j).astype(np.int64) * self.nv + np.maximum(i, j)
        return np.searchsorted(self.keys, k)

    def gradient(self):
        """Discrete gradient G (edges x vertices): (G phi)_e = phi_j - phi_i."""
        e = np.arange(self.n)
        rows = np.concatenate([e, e])
        cols = np.concatenate([self.edges[:, 0], self.edges[:, 1]])
        vals = np.concatenate([-np.ones(self.n), np.ones(self.n)])
        return sp.csr_matrix((vals, (rows, cols)), shape=(self.n, self.nv))


def _local_order(tets):
    """For each tet and local edge, the local vertex numbers (i,j) with global(i) < global(j)."""
    a, b = _LOC_EDGES[:, 0], _LOC_EDGES[:, 1]
    swap = tets[:, a] > tets[:, b]
    li = np.where(swap, b, a)
    lj = np.where(swap, a, b)
    return li, lj


def _pair_terms(D, I, li, lj):
    """Integral of w_a . w_b for all edge pairs, given gradient Gram D (n,m,m) and vertex-pair integrals I."""
    n = D.shape[0]
    ar = np.arange(n)[:, None, None]
    i, j = li[:, :, None], lj[:, :, None]
    k, l = li[:, None, :], lj[:, None, :]
    return (D[ar, j, l] * I[ar, i, k] - D[ar, j, k] * I[ar, i, l]
            - D[ar, i, l] * I[ar, j, k] + D[ar, i, k] * I[ar, j, l])


def assemble_volume(verts, tets, et, eps=1.0):
    """Curl-curl K and eps-weighted mass M (eps scalar or per-tet array, may be complex)."""
    nt = len(tets)
    p = verts[tets]                                            # (nt,4,3)
    T = p[:, 1:, :] - p[:, :1, :]
    detT = np.linalg.det(T)
    V = np.abs(detT) / 6.0
    Tinv = np.linalg.inv(T)                                    # grad l_{1..3} = rows of inv(T)^T
    g = np.empty((nt, 4, 3))
    g[:, 1:, :] = np.transpose(Tinv, (0, 2, 1))
    g[:, 0, :] = -g[:, 1:, :].sum(axis=1)
    D = np.einsum("tia,tja->tij", g, g)
    li, lj = _local_order(tets)
    I = (V[:, None, None] / 20.0) * (1.0 + np.eye(4)[None])
    Mloc = _pair_terms(D, I, li, lj)
    ar = np.arange(nt)[:, None, None]
    i, j = li[:, :, None], lj[:, :, None]
    k, l = li[:, None, :], lj[:, None, :]
    Kloc = 4.0 * V[:, None, None] * (D[ar, i, k] * D[ar, j, l] - D[ar, i, l] * D[ar, j, k])
    eps = np.broadcast_to(np.asarray(eps), (nt,))
    Mloc = Mloc * eps[:, None, None]
    rows = np.repeat(et.tet_edges[:, :, None], 6, axis=2).ravel()
    cols = np.repeat(et.tet_edges[:, None, :], 6, axis=1).ravel()
    n = et.n
    K = sp.csr_matrix((Kloc.ravel(), (rows, cols)), shape=(n, n))
    M = sp.csr_matrix((Mloc.ravel(), (rows, cols)), shape=(n, n))
    return K, M


def boundary_faces(verts, tets, lo, hi, tol=1e-12):
    """Triangles of tets lying entirely on a face of the box [lo,hi]."""
    tris = tets[:, _LOC_FACES].reshape(-1, 3)
    P = verts[tris]
    on = np.zeros(len(tris), bool)
    for ax in range(3):
        for bound in (lo[ax], hi[ax]):
            on |= np.all(np.abs(P[:, :, ax] - bound) < tol, axis=1)
    return tris[on]


def assemble_boundary_mass(verts, tris, et):
    """Tangential mass on boundary triangles (Silver-Mueller / Leontovich term)."""
    nt = len(tris)
    p = verts[tris]
    E = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]], axis=2)   # (nt,3,2)
    Gm = np.einsum("tai,taj->tij", E, E)
    area = 0.5 * np.sqrt(np.linalg.det(Gm))
    g12 = np.einsum("tai,tij->taj", E, np.linalg.inv(Gm))          # (nt,3,2)
    g = np.concatenate([-g12.sum(axis=2, keepdims=True), g12], axis=2).transpose(0, 2, 1)  # (nt,3,3)
    D = np.einsum("tia,tja->tij", g, g)
    la, lb = np.array([0, 0, 1]), np.array([1, 2, 2])
    swap = tris[:, la] > tris[:, lb]
    li, lj = np.where(swap, lb, la), np.where(swap, la, lb)
    I = (area[:, None, None] / 12.0) * (1.0 + np.eye(3)[None])
    Bloc = _pair_terms(D, I, li, lj)
    gi, gj = tris[np.arange(nt)[:, None], li], tris[np.arange(nt)[:, None], lj]
    eid = et.lookup(gi, gj)
    rows = np.repeat(eid[:, :, None], 3, axis=2).ravel()
    cols = np.repeat(eid[:, None, :], 3, axis=1).ravel()
    return sp.csr_matrix((Bloc.ravel(), (rows, cols)), shape=(et.n, et.n))
