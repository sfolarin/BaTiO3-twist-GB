#!/usr/bin/env python3
"""
Grain-interior polarization components along the field cycle (reads analysis/cellP.npz).

For each run: mean Px, Py, Pz and mean |P| over grain-interior cells (GB cells excluded) at four
points of the cycle -- remanent (+), +Emax, remanent (-), -Emax -- plus loop metrics recomputed
from the INTERIOR P_par only (summary.csv currently averages over all cells, GB cells included).

  python analysis/components.py runs/04_sigma5/300K/s5_001_300K runs/04_sigma5/10K/s5_001_10K ...
"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from btoan.loops import loop_metrics


def pick(E, Emax):
    """frame indices: first E=+Emax, the E~0 after it (descending), first E=-Emax, E~0 after it."""
    ip = int(np.argmax(E > 0.999 * Emax)); im = int(np.argmax(E < -0.999 * Emax))
    r_minus = ip + int(np.argmin(np.abs(E[ip:im])))        # zero crossing on the way down -> P_r(+)
    r_plus = im + int(np.argmin(np.abs(E[im:])))           # zero crossing on the way up   -> P_r(-)
    return {"Pr+(E=0 desc)": r_minus, "+Emax": ip, "Pr-(E=0 asc)": r_plus, "-Emax": im}


for run in sys.argv[1:]:
    rd = Path(run); z = np.load(rd / "analysis" / "cellP.npz")
    P, E, n = z["P"], z["E"], z["n"]
    inner = z["grain1"] | z["grain2"]
    if not inner.any():
        inner = ~z["gb"]
    Emax = float(np.nanmax(np.abs(E)))
    print(f"\n== {rd.name}   n={np.round(n, 3)}   interior cells {inner.sum()}/{len(inner)}")
    print(f"{'state':16s} {'E':>7s} {'<Px>':>7s} {'<Py>':>7s} {'<Pz>':>7s} {'<|P|>':>7s} {'<P.n>':>7s}")
    for lab, i in pick(E, Emax).items():
        p = P[i][inner]
        print(f"{lab:16s} {E[i]:7.3f} {p[:,0].mean():7.3f} {p[:,1].mean():7.3f} {p[:,2].mean():7.3f} "
              f"{np.linalg.norm(p, axis=1).mean():7.3f} {(p @ n).mean():7.3f}")
    m = loop_metrics(E, (P[:, inner] @ n).mean(1), Emax)
    keep = ("Ec", "Pr", "Ps", "squareness", "P_offset", "Ec_imprint")
    print("interior loop:", json.dumps({k: round(float(m[k]), 4) for k in keep if k in m}))
