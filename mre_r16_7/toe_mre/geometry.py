"""
Geometry observable module for TOE R16.7.

This module does not claim to derive geometry from first principles.
It implements an auditable geometric observable:
- build the graph Laplacian of an N x N torus;
- calculate the heat trace K(t);
- estimate spectral dimension D_s(t) = -2 d log K / d log t.

For a planted 2D torus, an intermediate-scale estimate should be near 2.
"""

from __future__ import annotations

import numpy as np


def torus_laplacian_eigenvalues(n: int) -> np.ndarray:
    """
    Eigenvalues of the combinatorial Laplacian for an N x N periodic square lattice.

    lambda(kx, ky) = 4 - 2 cos(2pi kx/N) - 2 cos(2pi ky/N)
    """
    if n < 3:
        raise ValueError("n must be >= 3 for the spectral geometry check")
    ks = np.arange(n)
    vals = []
    for kx in ks:
        for ky in ks:
            lam = (
                4.0
                - 2.0 * np.cos(2.0 * np.pi * kx / n)
                - 2.0 * np.cos(2.0 * np.pi * ky / n)
            )
            vals.append(lam)
    vals = np.array(vals, dtype=float)
    vals.sort()
    return vals


def heat_trace(eigenvalues: np.ndarray, times: np.ndarray, normalized: bool = True) -> np.ndarray:
    """
    Heat trace K(t) = sum exp(-lambda_i t).
    If normalized=True, divide by number of eigenvalues.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    times = np.asarray(times, dtype=float)
    if np.any(times <= 0):
        raise ValueError("all times must be positive")
    k = np.exp(-np.outer(times, eigenvalues)).sum(axis=1)
    if normalized:
        k = k / len(eigenvalues)
    return k


def spectral_dimension_curve(eigenvalues: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Estimate D_s(t) = -2 d log K(t) / d log t using finite differences.
    """
    k = heat_trace(eigenvalues, times, normalized=True)
    logk = np.log(k)
    logt = np.log(times)
    slope = np.gradient(logk, logt)
    return -2.0 * slope


def spectral_dimension_summary(n: int = 32) -> dict:
    """
    Return a robust intermediate-scale dimension estimate for a planted 2D torus.
    """
    eigenvalues = torus_laplacian_eigenvalues(n)
    # Avoid very small t (lattice UV) and very large t (finite-size IR).
    times = np.geomspace(1.0, 10.0, 32)
    ds = spectral_dimension_curve(eigenvalues, times)

    # Median in central window is more stable than a single point.
    central = ds[8:24]
    estimate = float(np.median(central))
    return {
        "n": n,
        "times_min": float(times.min()),
        "times_max": float(times.max()),
        "spectral_dimension_median_central": estimate,
        "spectral_dimension_min_central": float(np.min(central)),
        "spectral_dimension_max_central": float(np.max(central)),
        "expected_dimension_for_planted_torus": 2.0,
        "within_broad_tolerance": bool(1.5 <= estimate <= 2.5),
        "interpretation": "Pipeline sanity check: planted 2D torus returns dimension near 2.",
    }