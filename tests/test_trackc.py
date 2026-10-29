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
