"""Synthetic bowtie-like test problem: graded box, PEC bowtie halves plus wire tip, Si half-space, impedance outer boundary.

Units are dimensionless (box edge ~ 1). A(k) = K - k^2 M_eps - i k B, complex symmetric, PEC edges eliminated.
"""
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from . import mesh, nedelec


@dataclass
class Problem:
    K: sp.csr_matrix
    M: sp.csr_matrix
    B: sp.csr_matrix
    G: sp.csr_matrix          # discrete gradient on free edges, zero columns dropped
    b: np.ndarray
    probe: np.ndarray         # free-edge indices used as scalar output (gap field)
    h_min: float
    n_dof: int

    def A(self, k):
        return (self.K - k * k * self.M - 1j * k * self.B).tocsr()


def build(n_half=10, power=3.0, half_len=0.5, eps_si=11.7, absorber=0.2, absorb_strength=4.0):
    ax = np.linspace(-1.0, 1.0, 2 * n_half + 1)
    verts, tets = mesh.kuhn_mesh(ax, ax, ax)
    verts = mesh.radial_grade(verts, power, half_len)
    h0 = half_len * (1.0 / n_half) ** power
    et = nedelec.EdgeTable(tets, len(verts))
    cz = verts[tets][:, :, 2].mean(axis=1)
    eps = np.where(cz < 0, eps_si, 1.0).astype(complex)
    # crude absorbing shell (stand-in for the PML): quadratic loss ramp over the outer `absorber` fraction
    c = verts[tets].mean(axis=1)
    d = half_len - np.abs(c).max(axis=1)
    t = absorber * half_len
    eps = eps * (1 + 1j * absorb_strength * np.clip((t - d) / t, 0, None) ** 2)
    K, M = nedelec.assemble_volume(verts, tets, et, eps)
    lo, hi = np.full(3, -half_len), np.full(3, half_len)
    B = nedelec.assemble_boundary_mass(verts, nedelec.boundary_faces(verts, tets, lo, hi), et)

    # PEC: two bowtie halves in z=0 plane (gap at |x|<=h0), plus a wire tip above the gap.
    x, y, z = verts.T
    arm = 0.25 * half_len
    plate = (np.abs(z) < 1e-14) & (np.abs(x) > h0 * 0.5) & (np.abs(x) <= arm) & (np.abs(y) <= 0.5 * np.abs(x))
    wire = (np.abs(x) < 1e-14) & (np.abs(y) < 1e-14) & (z > 0) & (z <= arm)
    pec_v = plate | wire
    e = et.edges
    free = ~(pec_v[e[:, 0]] & pec_v[e[:, 1]])
    # gap edges: free x-directed edges at z=0, y=0 spanning the gap -> source and probe
    gap = free & (np.abs(z[e[:, 0]]) < 1e-14) & (np.abs(z[e[:, 1]]) < 1e-14) \
        & (np.abs(y[e[:, 0]]) < 1e-14) & (np.abs(y[e[:, 1]]) < 1e-14) \
        & (np.abs(x[e[:, 0]]) <= h0 * 1.01) & (np.abs(x[e[:, 1]]) <= h0 * 1.01)
    G = et.gradient().tocsr()[free]
    # Nodal space: H1 with phi = 0 on the PEC conductors, so that K G = 0 holds exactly on the free
    # edges (the de Rham exactness AMS relies on). This also removes the global constant.
    touched = np.diff(G.tocsc().indptr) > 0
    keep = np.flatnonzero(touched & ~pec_v)
    G = G[:, keep]
    fidx = np.cumsum(free) - 1
    sel = lambda Mat: Mat.tocsr()[free][:, free]
    b = np.zeros(int(free.sum()), complex)
    gap_free = fidx[gap]
    b[gap_free] = 1.0
    return Problem(sel(K), sel(M), sel(B), G.tocsr(), b, gap_free, h0, int(free.sum()))
