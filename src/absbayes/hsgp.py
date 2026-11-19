"""Hilbert-space Gaussian process (HSGP) basis for a 2D squared-exponential kernel.

A full GP over ~100k pitches is infeasible, so the surface is approximated with
M = m*m low-rank basis functions (Solin & Sarkka 2020; Riutort-Mayol et al. 2023).
The basis is computed here once; Stan only scales it by the spectral density, which
depends on the kernel's magnitude (alpha) and length-scale (rho).
"""
import numpy as np
import scipy.sparse as sp


def make_basis(x, z, L, m):
    """Eigenfunctions of the Laplacian on the box [-L, L]^2 (Dirichlet boundary).

    Returns
    -------
    Phi : (N, m*m) array of basis values at the points (x, z)
    omega : (m*m, 2) array of square-root eigenvalues for each basis function
    Column index is i*m + k for frequencies (omega_i, omega_k).
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    j = np.arange(1, m + 1)
    w = np.pi * j / (2.0 * L)
    phi_x = np.sqrt(1.0 / L) * np.sin(w[None, :] * (x[:, None] + L))
    phi_z = np.sqrt(1.0 / L) * np.sin(w[None, :] * (z[:, None] + L))
    Phi = (phi_x[:, :, None] * phi_z[:, None, :]).reshape(len(x), m * m)
    omega = np.array([(a, b) for a in w for b in w])
    return Phi, omega


def spectral_density_se(alpha, rho, omega):
    """Spectral density of the 2D squared-exponential kernel at each frequency pair."""
    sq = np.sum(np.asarray(omega) ** 2, axis=1)
    return alpha**2 * 2.0 * np.pi * rho**2 * np.exp(-0.5 * rho**2 * sq)


def approx_kernel(x, z, alpha, rho, L, m):
    """HSGP approximation of the SE kernel matrix, used for unit tests."""
    Phi, omega = make_basis(x, z, L, m)
    S = spectral_density_se(alpha, rho, omega)
    return (Phi * S[None, :]) @ Phi.T


def bilinear_matrix(x, z, lim, n):
    """Sparse (N, n*n) matrix W such that W @ f_grid bilinearly interpolates the surface.

    The grid is the regular n-by-n grid on [-lim, lim]^2 ordered like
    np.meshgrid(xs, zs).ravel() (row index = iz * n + ix). Each pitch touches only 4 grid
    values, so the per-pitch cost in Stan is 4 multiplies instead of one per basis function.
    """
    x = np.asarray(x, float)
    z = np.asarray(z, float)
    h = 2.0 * lim / (n - 1)
    fx = np.clip((x + lim) / h, 0.0, n - 1 - 1e-9)
    fz = np.clip((z + lim) / h, 0.0, n - 1 - 1e-9)
    ix, iz = fx.astype(int), fz.astype(int)
    tx, tz = fx - ix, fz - iz
    rows = np.repeat(np.arange(len(x)), 4)
    cols = np.stack([iz * n + ix, iz * n + ix + 1,
                     (iz + 1) * n + ix, (iz + 1) * n + ix + 1], axis=1).ravel()
    vals = np.stack([(1 - tx) * (1 - tz), tx * (1 - tz),
                     (1 - tx) * tz, tx * tz], axis=1).ravel()
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(x), n * n))
