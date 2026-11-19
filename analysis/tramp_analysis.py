#!/usr/bin/env python3
"""
Phase sequence from temperature-ramp runs (templates/tramp.in).

Per dump frame:
  * per-cell polarization; its three components sorted by magnitude, averaged over cells
    (|P|_1 >= |P|_2 >= |P|_3). These stay finite in the cubic phase (local off-centring,
    order-disorder) -> use them for R/O/T, not for C.
  * net polarization |<P>| (mean vector over all cells): long-range order, ~0 in the cubic phase.
  * pseudo-cubic lattice: cell edges / n (n unit cells per edge) sorted a <= b <= c, and the angles.
Phase per frame from the lattice (5-frame running median of the sorted edges), threshold dthr:
  T: c-b > dthr, b-a < dthr      O: b-a > dthr, c-b < dthr
  R: all within dthr and |P|_3 > p3thr      C: all within dthr and |P|_3 <= p3thr
Transitions are reported where the phase changes and the new phase persists >= `persist` frames.

  python analysis/tramp_analysis.py runs/01_benchmark/tramp_heat runs/01_benchmark/tramp_cool
-> <run>/analysis/tramp.csv, transitions printed, figure in the FIRST run's analysis/tramp.png
"""
import sys, argparse, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from btoan.io import read_data, iter_dump
from btoan.polarization import build_cells, cell_polarization

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--dthr", type=float, default=0.025, help="edge-splitting threshold, A per unit cell")
ap.add_argument("--p3thr", type=float, default=0.10, help="|P|_3 above which an equal-edge cell is R")
ap.add_argument("--persist", type=int, default=5)
a = ap.parse_args()


def run_median(x, w=5):
    h = w // 2
    return np.array([np.median(x[max(0, i - h):i + h + 1], axis=0) for i in range(len(x))])


def classify(edges, p3):
    lo, mid, hi = edges.T
    s_hi, s_lo = hi - mid, mid - lo
    ph = np.full(len(edges), "?", dtype=object)
    ph[(s_hi > a.dthr) & (s_lo <= a.dthr)] = "T"
    ph[(s_lo > a.dthr) & (s_hi <= a.dthr)] = "O"
    eq = (s_hi <= a.dthr) & (s_lo <= a.dthr)
    ph[eq & (p3 > a.p3thr)] = "R"
    ph[eq & (p3 <= a.p3thr)] = "C"
    return ph


def transitions(T, ph):
    out, cur = [], None
    for i in range(len(ph)):
        if ph[i] == "?" or ph[i] == cur:
            continue
        if all(p == ph[i] for p in ph[i:i + a.persist]) and len(ph[i:i + a.persist]) == a.persist:
            if cur is not None:
                out.append((cur, ph[i], T[i]))
            cur = ph[i]
    return out


fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for run in a.runs:
    rd = Path(run); out = rd / "analysis"; out.mkdir(exist_ok=True)
    Tmap = {}
    for l in open(rd / "traj_T.dat"):
        if not l.startswith("#"):
            p = l.split(); Tmap[int(p[0])] = float(p[1])
    D = read_data(rd / "input.data")
    cells = build_cells(D["type"], D["q"], D["x"], D["box"], D.get("csid"))
    n = round(len(cells["ti"]) ** (1 / 3))                      # unit cells per edge (cubic supercell)
    rows = []
    for fr in iter_dump(rd / "dump_T.lammpstrj"):
        pc = cell_polarization(cells, fr["x"], fr["H"])
        comp = np.sort(np.abs(pc), axis=1)[:, ::-1].mean(0)
        net = pc.mean(0)
        H = fr["H"]; L = [np.linalg.norm(H[:, i]) for i in range(3)]
        ang = lambda u, v: np.degrees(np.arccos(H[:, u] @ H[:, v] / L[u] / L[v]))
        rows.append([fr["step"], Tmap.get(fr["step"], np.nan), *comp, np.linalg.norm(pc, axis=1).mean(),
                     np.linalg.norm(net), *(np.array(L) / n), ang(1, 2), ang(0, 2), ang(0, 1)])
    R = np.array(rows)
    # nominal (thermostat target) temperature from the protocol, if params.json is present:
    # instantaneous T of a 17k-particle cell scatters by several K and scrambles the curve order
    try:
        v = json.load(open(rd / "params.json"))["lammps_vars"]
        T0, T1, nr, nh = (float(v[k]) for k in ("T0", "T1", "nramp", "nhold"))
        R[:, 1] = np.where(R[:, 0] < nh, T0, np.where(R[:, 0] < nh + nr, T0 + (T1 - T0) * (R[:, 0] - nh) / nr, T1))
        print(f"{rd.name}: using nominal ramp temperature {T0:.0f} -> {T1:.0f} K")
    except (FileNotFoundError, KeyError):
        print(f"{rd.name}: no params.json -> using instantaneous T")
    R = R[np.isfinite(R[:, 1])]
    edges = run_median(np.sort(R[:, 7:10], axis=1))
    ph = classify(edges, run_median(R[:, 4]))
    np.savetxt(out / "tramp.csv", np.column_stack([R, edges]), delimiter=",", comments="",
               header="step,T,P1,P2,P3,absP,netP,a,b,c,alpha,beta,gamma,edge_lo,edge_mid,edge_hi")
    tr = transitions(R[:, 1], ph)
    print(f"{rd.name}: n={n}  dthr={a.dthr} A  p3thr={a.p3thr}")
    print("  phases seen:", "".join(p for i, p in enumerate(ph) if i == 0 or p != ph[i - 1]))
    for p0, p1, T in tr:
        print(f"  {p0} -> {p1} at T = {T:.0f} K")
    lab = rd.name
    for k, ls in zip((2, 3, 4), ("-", "--", ":")):
        ax[0].plot(R[:, 1], R[:, k], ls, label=f"{lab} |P|_{k-1}")
    ax[1].plot(R[:, 1], R[:, 6], label=f"{lab}")
    for j in range(3):
        ax[2].plot(R[:, 1], edges[:, j], ".", ms=3, label=lab if j == 0 else None)
    for p0, p1, T in tr:
        for x in ax:
            x.axvline(T, color="0.8", lw=0.6, zorder=0)
ax[0].set(xlabel="T (K)", ylabel="sorted per-cell |P| components (C/m²)", title="local polarization (R/O/T)")
ax[0].legend(fontsize=7)
ax[1].set(xlabel="T (K)", ylabel="net |⟨P⟩| (C/m²)", title="long-range polarization (C: ~0)"); ax[1].legend(fontsize=7)
ax[2].set(xlabel="T (K)", ylabel="pseudo-cubic edge (Å per unit cell)", title="lattice (5-frame median, sorted)")
ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(Path(a.runs[0]) / "analysis" / "tramp.png", dpi=150)
print("wrote tramp.csv / tramp.png")
