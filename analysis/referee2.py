#!/usr/bin/env python3
"""
Second-round referee checks that use EXISTING data only.

  grains   (major 1) per-grain polarization at the remanent states, in the lab frame AND in each grain's own
           crystal frame: mean components, the occupied <111>/<110>/<100> variants, and the R variant that
           would maximise P_z with P_y < 0. Tells whether the 10 K bicrystal sits in the best-projecting
           R variant and, if not, what constrains it (P_x of the two grains, |P|, O-like states).
             python analysis/referee2.py grains runs/04_sigma5/10K/s5_001_10K runs/06_finite_size/10K/s5L_001_10K \
                 runs/05_sigma61/10K/s61_001_10K runs/04_sigma5/10K/s5r_001_10K \
                 runs/03_single_crystal/10K/sc_m102_10K runs/03_single_crystal/10K/sc_m506_10K

Frames used: the instantaneous frames at E = 0 after +Emax (P_r+) and after -Emax (P_r-), plus the frame before each
(field 0.002 V/A, same branch). Interior cells only (stored grain masks). Writes figures/out/referee_grains.json.
"""
import json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "out"; OUT.mkdir(parents=True, exist_ok=True)
P_R = 0.273                                   # |P| of the R phase at 0 K (Table S2)


def pick(E):
    Em = np.nanmax(np.abs(E)); ip = int(np.argmax(E > 0.999 * Em)); im = int(np.argmax(E < -0.999 * Em))
    return {"Pr+": ip + int(np.argmin(np.abs(E[ip:im]))), "Pr-": im + int(np.argmin(np.abs(E[im:])))}


def frames_of(run):
    """lab unit vectors (rows x,y,z) expressed in crystal coordinates, per grain."""
    prm = json.load(open(run / "params.json")) if (run / "params.json").exists() else {}
    st = prm.get("structure", run.name)
    if st.startswith("sc"):
        return {"crystal": np.eye(3)}
    m = re.search(r"_(\d)(\d)0", st)
    if m:
        h, k = int(m.group(1)), int(m.group(2))
    elif run.name.startswith("s61"):
        h, k = 6, 5
    elif run.name.startswith("s5"):
        h, k = 2, 1
    else:
        return {"crystal": np.eye(3)}
    out = {}
    for g, s in (("grain1", +1), ("grain2", -1)):
        x = np.array([h, 0, s * k], float); x /= np.linalg.norm(x); y = np.array([0, 1.0, 0]); z = np.cross(x, y)
        out[g] = np.vstack([x, y, z])
    if "s5r" in st or "rev" in st:
        out["note"] = "reversed stacking (same grain orientations)"
    return out


def variant(pc, thr=0.06):
    s = np.where(pc > thr, "+", np.where(pc < -thr, "-", "0"))
    return ["[" + "".join(r) + "]" for r in s]


def best_R(Rm):
    """R variant (crystal <111>) with lab P_y < 0 that maximises lab P_z; returned in lab components (C/m^2)."""
    best = None
    for a in (-1, 1):
        for b in (-1, 1):
            for c in (-1, 1):
                v = np.array([a, b, c]) / np.sqrt(3) * P_R; lab = Rm @ v
                if lab[1] < 0 and (best is None or lab[2] > best[1][2]):
                    best = (variant(v[None])[0], lab)
    return best


def grains(runs):
    res = {}
    for run in map(Path, runs):
        z = np.load(run / "analysis" / "cellP.npz"); P, E = z["P"], z["E"]
        fr = frames_of(run); k = pick(E)
        if "crystal" in fr or not z["grain1"].any():
            masks = {"crystal": np.ones(P.shape[1], bool)}
        else:
            masks = {"grain1": z["grain1"], "grain2": z["grain2"]}
        res[run.name] = {}
        print(f"\n== {run.name}")
        for state, i in k.items():
            idx = [max(0, i - 1), i]
            for g, m in masks.items():
                p = np.concatenate([P[j][m] for j in idx])               # instantaneous, two frames pooled
                Rm = fr[g]                                                 # rows = lab axes in crystal coords
                pc = p @ Rm                                                # crystal components: sum_l P_l * e_l
                mag = np.sort(np.abs(pc), axis=1)[:, ::-1]
                vs, cnt = np.unique(variant(pc), return_counts=True); o = np.argsort(cnt)[::-1][:4]
                bv, blab = best_R(Rm)                                      # crystal -> lab: Rm @ v
                row = dict(P_lab=p.mean(0).round(4).tolist(), absP=round(float(np.linalg.norm(p, axis=1).mean()), 4),
                           sorted_crystal_components=mag.mean(0).round(4).tolist(),
                           variants={str(vs[j]): round(float(cnt[j] / len(p)), 3) for j in o},
                           best_R_variant=bv, best_R_lab=blab.round(4).tolist())
                res[run.name][f"{state} {g}"] = row
                print(f"{state:4s} {g:8s} <P>lab=({row['P_lab'][0]:+.3f},{row['P_lab'][1]:+.3f},{row['P_lab'][2]:+.3f}) "
                      f"<|P|>={row['absP']:.3f}  |P|1,2,3=({mag.mean(0)[0]:.3f},{mag.mean(0)[1]:.3f},{mag.mean(0)[2]:.3f})  "
                      f"variants {row['variants']}")
                if state == "Pr+":
                    print(f"{'':13s} best R variant with Py<0: {bv} -> lab ({blab[0]:+.3f},{blab[1]:+.3f},{blab[2]:+.3f})")
    json.dump(res, open(OUT / "referee_grains.json", "w"), indent=1)
    print("\nwrote figures/out/referee_grains.json")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "grains":
        sys.exit(__doc__)
    grains(sys.argv[2:])
