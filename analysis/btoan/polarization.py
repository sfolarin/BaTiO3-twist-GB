"""
Per-unit-cell polarization for the BaTiO3 core-shell model.

Each Ti defines one perovskite cell: the Ti (weight 1), its 6 nearest O (weight 1/2)
and 8 nearest Ba (weight 1/8). Cores and shells enter with their own charges:

    p_cell = sum_i w_i q_i (r_i - r_Ti_core),     P_cell = p_cell / V_cell

The weighted charge of the cell is exactly zero (3.072 - 6*0.5*1.748 + 8*0.125*2.172 = 0),
so P_cell does not depend on the origin or on where the periodic box is cut -- unlike the
whole-cell 'compute dipole', which carries an arbitrary offset of ~0.2 C/m^2 for these cells.
The cell-averaged P_cell equals the bulk polarization (checked in tests/test_polarization.py).

Topology (which O and Ba belong to which Ti) is built once from a reference frame and kept,
so it is unaffected by large ionic displacements during switching.
"""
import numpy as np
from scipy.spatial import cKDTree

E_A3_TO_C_M2 = 16.0218
WEIGHTS = {2: 1.0, 3: 0.5, 1: 0.125}  # by core type: Ti, O, Ba


def _mic(d, box):
    """Minimum image for an orthogonal box (box = 3 lengths) or a triclinic cell (box = 3x3 H, columns = cell vectors)."""
    box = np.asarray(box, float)
    if box.ndim == 1:
        return d - np.round(d / box) * box
    Hinv = np.linalg.inv(box)
    s = d @ Hinv.T
    return (s - np.round(s)) @ box.T


def build_cells(types, q, x_ref, box, csid=None):
    """Return a dict describing every Ti-centred cell.
    types/q/x_ref are per particle (cores types 1-3, shells 4-6), sorted by id.
    Shell of a core is found via csid (same CS id) or, if None, assumed to be the next particle."""
    n = len(types)
    core = np.where(types <= 3)[0]
    if csid is not None:
        first = {}
        shell_of = np.full(n, -1)
        for i in np.argsort(csid, kind="stable"):
            c = csid[i]
            if c in first:
                a, b = first[c], i
                if types[a] <= 3: shell_of[a] = b
                else: shell_of[b] = a
            else:
                first[c] = i
    else:
        shell_of = np.full(n, -1); shell_of[core] = core + 1
    if np.any(shell_of[core] < 0):
        raise ValueError("core without shell")
    xw = np.mod(x_ref, box)
    ti = core[types[core] == 2]
    o = core[types[core] == 3]
    ba = core[types[core] == 1]
    t_o = cKDTree(xw[o], boxsize=box); t_ba = cKDTree(xw[ba], boxsize=box)
    d_o, j_o = t_o.query(xw[ti], k=6)
    d_ba, j_ba = t_ba.query(xw[ti], k=8)
    members = np.concatenate([ti[:, None], o[j_o], ba[j_ba]], axis=1)          # (nTi, 15) core indices
    w = np.concatenate([np.ones((len(ti), 1)), 0.5 * np.ones((len(ti), 6)), 0.125 * np.ones((len(ti), 8))], axis=1)
    # distortion flag: cells whose cage is irregular in the reference (GB cores)
    irregular = (d_o.max(1) - d_o.min(1) > 0.35) | (d_ba.max(1) - d_ba.min(1) > 0.5)
    qc = q[members]; qs = q[shell_of[members]]
    net = (w * (qc + qs)).sum(1)
    return dict(ti=ti, members=members, shells=shell_of[members], w=w, qcore=qc, qshell=qs,
                irregular=irregular, net_charge=net, ti_ref=xw[ti])


def cell_polarization(cells, x, box):
    """P_cell (nTi, 3) in C/m^2 for one frame (x unwrapped or wrapped, sorted like the reference).
    box: 3 edge lengths (orthogonal) or the 3x3 cell matrix H (triclinic)."""
    ti = cells["ti"]
    r0 = x[ti][:, None, :]
    dc = _mic(x[cells["members"]] - r0, box)
    ds = _mic(x[cells["shells"]] - r0, box)
    p = (cells["w"][..., None] * (cells["qcore"][..., None] * dc + cells["qshell"][..., None] * ds)).sum(1)
    box = np.asarray(box, float)
    vcell = (np.prod(box) if box.ndim == 1 else abs(np.linalg.det(box))) / len(ti)
    return p / vcell * E_A3_TO_C_M2


def profile_along(values, coord, box_len, nbins):
    """Bin a per-cell quantity (n, k) along one coordinate -> (centers, mean (nbins,k), count)."""
    edges = np.linspace(0, box_len, nbins + 1)
    idx = np.clip(np.digitize(np.mod(coord, box_len), edges) - 1, 0, nbins - 1)
    out = np.zeros((nbins, values.shape[1])); cnt = np.bincount(idx, minlength=nbins)
    for k in range(values.shape[1]):
        out[:, k] = np.bincount(idx, weights=values[:, k], minlength=nbins) / np.maximum(cnt, 1)
    return 0.5 * (edges[1:] + edges[:-1]), out, cnt
