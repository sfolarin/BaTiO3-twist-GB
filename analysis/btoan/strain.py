"""
Grain-resolved strain from atomic positions.

For each grain we fit the affine map  x - <x> = F (X - <X>)  over the Ti cores in the grain
INTERIOR (a margin away from both periodic GBs), using unwrapped coordinates. The small-strain
tensor is eps = (F + F^T)/2 - I, in the lab frame. The strain along the field is n.eps.n.

This replaces the old 'grain strain', which was algebraically identical to the global box
strain (xx, zz) or identically zero (yy).
"""
import numpy as np


def grain_masks(y_frac_ref, gb_planes_frac=(0.0, 0.5), margin_frac=0.05):
    """Two grains between GB planes at fractional y = 0 and 0.5 (periodic bicrystal).
    Returns boolean masks (grain1: 0..0.5, grain2: 0.5..1) excluding +-margin around both GBs."""
    y = np.mod(y_frac_ref, 1.0)
    g1 = (y > gb_planes_frac[0] + margin_frac) & (y < gb_planes_frac[1] - margin_frac)
    g2 = (y > gb_planes_frac[1] + margin_frac) & (y < 1.0 - margin_frac)
    return g1, g2


def fit_F(X, x):
    """Least-squares deformation gradient mapping reference X -> current x (both (n,3), unwrapped)."""
    A = X - X.mean(0); B = x - x.mean(0)
    Ft, *_ = np.linalg.lstsq(A, B, rcond=None)   # B = A Ft  ->  F = Ft^T
    return Ft.T


def small_strain(F):
    return 0.5 * (F + F.T) - np.eye(3)


def strain_along(eps, n):
    n = np.asarray(n, float); n = n / np.linalg.norm(n)
    return float(n @ eps @ n)


def box_strain_along(L, L0, n):
    """Global strain along n from orthogonal box lengths (aniso barostat: no shear)."""
    n = np.asarray(n, float); n = n / np.linalg.norm(n)
    e = (np.asarray(L) - L0) / L0
    return (e * n ** 2).sum(-1)
