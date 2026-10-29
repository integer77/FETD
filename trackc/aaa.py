"""AAA rational approximation (Nakatsukasa, Sete, Trefethen 2018) and an adaptive frequency sampler."""
import numpy as np


class AAA:
    def __init__(self, z, f, tol=1e-12, mmax=40):
        z, f = np.asarray(z, complex), np.asarray(f, complex)
        M = len(z)
        mask = np.ones(M, bool)
        zj, fj, wj = [], [], None
        R = np.full(M, f.mean())
        for m in range(min(mmax, M)):
            j = np.argmax(np.where(mask, np.abs(f - R), -1))
            zj.append(z[j]); fj.append(f[j]); mask[j] = False
            if not mask.any():
                break
            C = 1.0 / (z[mask, None] - np.array(zj)[None, :])
            Af = (f[mask, None] - np.array(fj)[None, :]) * C
            _, _, Vh = np.linalg.svd(Af, full_matrices=True)
            wj = Vh[-1].conj()
            R = f.copy()
            N = C @ (wj * np.array(fj)); D = C @ wj
            R[mask] = N / D
            if np.max(np.abs(f - R)) <= tol * np.max(np.abs(f)):
                break
        self.zj, self.fj, self.wj = np.array(zj), np.array(fj), wj if wj is not None else np.ones(len(zj))

    def __call__(self, x):
        x = np.atleast_1d(np.asarray(x, complex))
        d = x[:, None] - self.zj[None, :]
        hit = np.isclose(d, 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            C = 1.0 / d
            r = (C @ (self.wj * self.fj)) / (C @ self.wj)
        for i, j in zip(*np.nonzero(hit)):
            r[i] = self.fj[j]
        return r


def adaptive_sample(f, lo, hi, tol=1e-3, n0=5, nmax=40, ngrid=400):
    """Sample scalar function f (complex valued, e.g. a probe output at wavenumber k) on [lo, hi].

    Adds the point where two AAA models (all samples vs. all but the newest) disagree most, until the
    disagreement is below tol relative to max|f|. Returns (ks, values, model).
    """
    ks = list(np.linspace(lo, hi, n0))
    vals = [f(k) for k in ks]
    grid = np.linspace(lo, hi, ngrid)
    while True:
        full = AAA(ks, vals, tol=tol * 1e-2)
        part = AAA(ks[:-1], vals[:-1], tol=tol * 1e-2)
        diff = np.abs(full(grid) - part(grid))
        scale = np.max(np.abs(vals))
        if len(ks) >= nmax or diff.max() < tol * scale:
            return np.array(ks), np.array(vals), full
        taken = np.array(ks)
        diff[np.min(np.abs(grid[:, None] - taken[None, :]), axis=1) < (hi - lo) / ngrid] = 0
        k_new = grid[np.argmax(diff)]
        ks.append(k_new); vals.append(f(k_new))
