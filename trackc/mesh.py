"""Graded Kuhn-tetrahedral box mesh, the stand-in for the gmsh bowtie mesh."""
import numpy as np


def radial_grade(verts, power, half_len):
    """Map a uniform grid on [-1,1]^3 to p' = half_len * p * r^(power-1), r = |p|_inf.

    Keeps cells close to isotropic: the finest spacing at the centre is half_len * (1/n_half)^power,
    cell size grows smoothly (no needle cells, unlike a tensor-product grading).
    """
    r = np.abs(verts).max(axis=1, keepdims=True)
    return half_len * verts * np.where(r > 0, r, 1.0) ** (power - 1)


def kuhn_mesh(xs, ys, zs):
    """Return (verts (nv,3), tets (nt,4)). All cells use the 000-111 diagonal, so the mesh conforms."""
    nx, ny, nz = len(xs), len(ys), len(zs)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    idx = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    base = idx[:-1, :-1, :-1].ravel()
    off = np.array([ny * nz, nz, 1])
    tets = []
    for perm in ([0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]):
        v1 = base + off[perm[0]]
        v2 = v1 + off[perm[1]]
        v3 = base + off.sum()
        tets.append(np.stack([base, v1, v2, v3], axis=1))
    return verts, np.concatenate(tets)
