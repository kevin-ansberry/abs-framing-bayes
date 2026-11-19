import numpy as np

from absbayes.hsgp import approx_kernel, bilinear_matrix, make_basis


def exact_se(x, z, alpha, rho):
    pts = np.column_stack([x, z])
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    return alpha**2 * np.exp(-0.5 * d2 / rho**2)


def test_basis_shapes():
    x = np.linspace(-2, 2, 7)
    Phi, omega = make_basis(x, x, L=3.0, m=5)
    assert Phi.shape == (7, 25) and omega.shape == (25, 2)


def test_hsgp_matches_exact_kernel_in_interior():
    rng = np.random.default_rng(0)
    x, z = rng.uniform(-2, 2, 80), rng.uniform(-2, 2, 80)
    # Domain half-width L must be well beyond the data (here 4 vs 2), otherwise the
    # Dirichlet boundary shrinks the kernel near the edges.
    approx = approx_kernel(x, z, alpha=1.0, rho=0.8, L=4.0, m=20)
    assert np.max(np.abs(approx - exact_se(x, z, 1.0, 0.8))) < 0.01


def test_boundary_effect_when_domain_too_tight():
    rng = np.random.default_rng(2)
    x, z = rng.uniform(-2, 2, 80), rng.uniform(-2, 2, 80)
    ex = exact_se(x, z, 1.0, 0.8)
    err_tight = np.max(np.abs(approx_kernel(x, z, 1.0, 0.8, 2.2, 20) - ex))
    err_wide = np.max(np.abs(approx_kernel(x, z, 1.0, 0.8, 4.0, 20) - ex))
    assert err_wide < err_tight


def test_rougher_surface_needs_more_basis_functions():
    rng = np.random.default_rng(1)
    x, z = rng.uniform(-2, 2, 60), rng.uniform(-2, 2, 60)
    ex = exact_se(x, z, 1.0, 0.3)
    err_small = np.max(np.abs(approx_kernel(x, z, 1.0, 0.3, 3.0, 6) - ex))
    err_big = np.max(np.abs(approx_kernel(x, z, 1.0, 0.3, 3.0, 24) - ex))
    assert err_big < err_small


def test_bilinear_rows_sum_to_one_and_interpolate_smooth_function():
    rng = np.random.default_rng(3)
    x, z = rng.uniform(-2.5, 2.5, 500), rng.uniform(-2.5, 2.5, 500)
    n, lim = 51, 2.5
    W = bilinear_matrix(x, z, lim, n)
    assert np.allclose(np.asarray(W.sum(axis=1)).ravel(), 1.0)
    gx, gz = np.meshgrid(np.linspace(-lim, lim, n), np.linspace(-lim, lim, n))
    f = lambda a, b: np.sin(a) * np.cos(b)
    approx = W @ f(gx, gz).ravel()
    assert np.max(np.abs(approx - f(x, z))) < 0.01


def test_bilinear_handles_boundary_points():
    W = bilinear_matrix(np.array([-2.5, 2.5]), np.array([2.5, -2.5]), 2.5, 51)
    assert np.allclose(np.asarray(W.sum(axis=1)).ravel(), 1.0)
