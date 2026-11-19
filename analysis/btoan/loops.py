"""
Hysteresis-loop metrics from the cycle stage of traj.dat. No centering: P is measured relative
to the as-built ideal structure, so the loop position is physical. Any residual vertical offset is
reported as 'P_offset' instead of being silently removed.

Branches: descending (+Emax -> -Emax) and ascending (-Emax -> +Emax) are taken from the
protocol segments, so Ec and Pr are read on the correct branch.
"""
import numpy as np


def _crossings(x, y, level=0.0):
    """x at which y crosses 'level' (linear interpolation)."""
    s = np.sign(y - level); idx = np.where(np.diff(s) != 0)[0]
    out = []
    for i in idx:
        y0, y1 = y[i] - level, y[i + 1] - level
        out.append(x[i] - y0 * (x[i + 1] - x[i]) / (y1 - y0) if y1 != y0 else x[i])
    return np.array(out)


def split_branches(E, Emax):
    """Indices of the descending and ascending full-span branches of the cycle."""
    i_top = int(np.argmax(E)); i_bot = int(np.argmin(E))
    # first +Emax -> following -Emax is descending; -Emax -> next +Emax is ascending
    i_top2 = i_bot + int(np.argmax(E[i_bot:]))
    return np.arange(i_top, i_bot + 1), np.arange(i_bot, i_top2 + 1)


def loop_metrics(E, P, Emax, smooth=1):
    if smooth > 1:
        k = np.ones(smooth) / smooth
        P = np.convolve(P, k, mode="same")
    dn, up = split_branches(E, Emax)
    Ec_dn = _crossings(E[dn], P[dn]); Ec_up = _crossings(E[up], P[up])
    Pr_dn = _crossings(P[dn], E[dn]); Pr_up = _crossings(P[up], E[up])
    def pick(a, f):  # outermost crossing if several (multistep loops)
        return float(f(a)) if len(a) else np.nan
    Ec_minus = pick(Ec_dn, np.min); Ec_plus = pick(Ec_up, np.max)
    Pr_plus = pick(Pr_dn, np.max); Pr_minus = pick(Pr_up, np.min)
    P_at_Emax = float(P[dn[0]]); P_at_mEmax = float(P[up[0]])
    return dict(
        Ec_plus=Ec_plus, Ec_minus=Ec_minus, Ec=0.5 * (Ec_plus - Ec_minus), Ec_imprint=0.5 * (Ec_plus + Ec_minus),
        Pr_plus=Pr_plus, Pr_minus=Pr_minus, Pr=0.5 * (Pr_plus - Pr_minus), P_offset=0.5 * (Pr_plus + Pr_minus),
        Ps=0.5 * (P_at_Emax - P_at_mEmax),
        n_steps_down=int(len(Ec_dn)), n_steps_up=int(len(Ec_up)),
        squareness=(0.5 * (Pr_plus - Pr_minus)) / (0.5 * (P_at_Emax - P_at_mEmax)),
        Emax=Emax)
