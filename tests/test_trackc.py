import numpy as np
import pytest
from trackc import problem, solver
from trackc.factor import Factorization
from trackc.sweep import FrequencySweep


@pytest.fixture(scope="module")
def prob():
    return problem.build(n_half=10, power=4.0)


def k_of(lam):
    return 2 * np.pi / lam


def test_matrix_is_complex_symmetric(prob):
    A = prob.A(k_of(1.0))
    assert abs(A - A.T).max() / abs(A).max() < 1e-12


def test_gradient_is_in_curl_nullspace(prob):
    # K G = 0 on free edges away from PEC rows (curl grad = 0), up to roundoff
    v = np.random.default_rng(0).standard_normal(prob.G.shape[1])
    g = prob.G @ v
    assert np.linalg.norm(prob.K @ g) < 1e-9 * np.linalg.norm(g) * abs(prob.K).max()


def test_plain_fp32_is_inaccurate_but_gradient_correction_fixes_it(prob):
    A = prob.A(k_of(1.0))
    xref = Factorization(A, np.complex128).solve(prob.b)
    F = Factorization(A)
    x32 = F.solve(prob.b)
    assert np.linalg.norm(x32 - xref) / np.linalg.norm(xref) > 1e-3     # plain FP32 is wrong
    res = solver.refine(A, prob.b, lambda r: F.solve(r), maxit=10)
    assert not res.converged                                            # classical IR does not rescue it
    P = solver.GradientCorrectedPrecond(A, prob.G, edge_factor=F)
    res = solver.refine(A, prob.b, P)
    assert res.converged and res.steps <= 10
    assert np.linalg.norm(res.x - xref) / np.linalg.norm(xref) < 1e-7


def test_fp32_nodal_solve_is_enough(prob):
    A = prob.A(k_of(1.0))
    P = solver.GradientCorrectedPrecond(A, prob.G, nodal_dtype=np.complex64)
    assert solver.refine(A, prob.b, P).converged


def test_recycled_factor_converges_at_nearby_frequency(prob):
    A0, A1 = prob.A(k_of(1.0)), prob.A(k_of(1.0) * 1.03)
    P = solver.GradientCorrectedPrecond(A0, prob.G)
    P.rebind(A1)
    xref = Factorization(A1, np.complex128).solve(prob.b)
    res = solver.gmres_ir(A1, prob.b, P)
    assert res.converged
    assert np.linalg.norm(res.x - xref) / np.linalg.norm(xref) < 1e-7


def test_sweep_refactors_only_when_needed(prob):
    sw = FrequencySweep(prob, reuse_tol=0.05)
    pts = sw.run(k_of(1.0) * np.array([1.0, 1.02, 1.04, 1.3]))
    assert [p.refactored for p in pts] == [True, False, False, True]
    assert all(p.converged for p in pts)
    assert sw.factorizations == 2


def test_general_api_with_affine_system_multiple_rhs(prob, tmp_path):
    from trackc import maxwell_system, MixedPrecisionSolver, io
    rng = np.random.default_rng(1)
    b = np.stack([prob.b, rng.standard_normal(prob.n_dof) + 0j], axis=1)
    sysm = maxwell_system(prob.K, prob.M, prob.B, prob.G, b)
    s = MixedPrecisionSolver(sysm)
    k = k_of(1.0)
    res = s.solve(k)
    assert res.converged and res.x.shape == b.shape
    A = sysm.A(k)
    assert np.linalg.norm(A @ res.x - b) / np.linalg.norm(b) < 1e-9
    res2 = s.solve(k * 1.02)
    assert res2.converged and s.factorizations == 1
    io.save_fixed(tmp_path / "s.npz", A, prob.G, prob.b)
    A2, G2, b2 = io.load_fixed(tmp_path / "s.npz")
    assert abs(A2 - A).max() == 0 and (G2 != prob.G).nnz == 0


def test_adaptive_aaa_resolves_a_resonance():
    from trackc.aaa import adaptive_sample
    f = lambda k: 1.0 / (k - (2.0 + 0.05j)) + 0.3 / (k - (3.1 + 0.2j)) + 0.1 * k
    ks, vals, model = adaptive_sample(f, 1.0, 4.0, tol=1e-4)
    grid = np.linspace(1.0, 4.0, 777)
    assert np.max(np.abs(model(grid) - f(grid))) < 2e-3 * np.max(np.abs(vals))
    assert len(ks) < 25
