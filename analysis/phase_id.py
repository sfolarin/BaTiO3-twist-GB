#!/usr/bin/env python3
"""
Phase identification frame by frame for single-crystal runs (temperature ramps and 300 K hysteresis runs).

Two independent measures per dump frame:
  1. Space group of the SPATIALLY AVERAGED 5-atom cell (spglib). Every core is mapped to its ideal site in the
     as-built cubic supercell; its displacement from that site (minimum image, in the current cell) is averaged
     over all cells for each of the five sublattices (Ba, Ti, O1, O2, O3). Averaging over ~10^3-10^4 cells removes
     thermal noise, so a tight tolerance can be used. The small-cell lattice is the supercell matrix / n.
       R3m (160) = rhombohedral, Amm2 (38) = orthorhombic, P4mm (99) = tetragonal, Pm-3m (221) = cubic.
     Valid for single-domain states; a multidomain state averages its displacements toward zero and reads as a
     higher symmetry than its domains (flagged by the ratio |<P>| / <|P|> printed per frame).
  2. Local character of each cell's polarization: the direction family it is closest to,
     <111> (R-like), <110> (O-like) or <100> (T-like); cells with |P| < --pmin are counted as 'small'.
     This is a local, order-disorder measure: in BaTiO3 cells stay off-centre along ~<111> even in the cubic phase.

  pip install --user spglib     (once, on NERSC)
  python analysis/phase_id.py runs/01_benchmark/tramp_heat runs/01_benchmark/tramp_cool
  python analysis/phase_id.py runs/03_single_crystal/300K/sc_001_300K ... --stride 5
-> <run>/analysis/phase_id.csv, a printed summary of the space-group windows, figures/out/phase_id.json
"""
import sys, json, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from btoan.io import read_data, iter_dump, read_traj
from btoan.polarization import build_cells, cell_polarization

try:
    import spglib
except ImportError:
    sys.exit("spglib not installed: pip install --user spglib")

ROOT = Path(__file__).resolve().parents[1]
SG_PHASE = {160: "R", 38: "O", 99: "T", 221: "C"}
ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--symprec", type=float, nargs="+", default=[0.01, 0.03, 0.05],
                help="spglib tolerances (A) for the averaged cell; the summary uses the middle one")
ap.add_argument("--pmin", type=float, default=0.05, help="cells with |P| below this (C/m^2) are 'small'")
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--persist", type=int, default=3, help="frames a new space group must persist to count")
a = ap.parse_args()


def families():
    d100 = np.eye(3)
    d110 = np.array([[1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]]) / np.sqrt(2)
    d111 = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]]) / np.sqrt(3)
    return [("T-like", d100), ("O-like", d110), ("R-like", d111)]


FAM = families()


def local_fractions(pc):
    mag = np.linalg.norm(pc, axis=1); big = mag >= a.pmin
    u = pc[big] / mag[big, None]
    best = np.stack([np.abs(u @ d.T).max(1) for _, d in FAM], axis=1)     # max |cos| per family
    k = best.argmax(1)
    out = {name: float((k == i).sum() / len(pc)) for i, (name, _) in enumerate(FAM)}
    out["small"] = float((~big).mean())
    return out


def setup(run):
    D = read_data(run / "input.data")
    box = np.asarray(D["box"], float); t = D["type"]
    core = np.where(t <= 3)[0]
    nTi = int((t == 2).sum())
    a0 = (np.prod(box) / nTi) ** (1 / 3)
    n = np.rint(box / a0).astype(int)
    if np.prod(n) != nTi:
        raise SystemExit(f"{run}: not a cubic-cell supercell (n={n}, nTi={nTi})")
    s_ref = (D["x"][core] - D["x"][core].min(0) * 0) / box                # fractional in the as-built box
    f = (s_ref * n) % 1.0
    ba0 = f[t[core] == 1][0]
    f = (f - ba0) % 1.0
    half = (np.abs(f - 0.5) < 0.25).astype(int)                           # 1 where a coordinate is ~1/2
    cls = np.full(len(core), -1)
    tc = t[core]
    cls[(tc == 1) & (half.sum(1) == 0)] = 0
    cls[(tc == 2) & (half.sum(1) == 3)] = 1
    for j in range(3):                                                    # O with coordinate j ~ 0: O(j)
        cls[(tc == 3) & (half.sum(1) == 2) & (half[:, j] == 0)] = 2 + j
    if (cls < 0).any():
        raise SystemExit(f"{run}: {int((cls < 0).sum())} cores could not be assigned to a sublattice")
    ideal = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]) + ba0
    cells = build_cells(D["type"], D["q"], D["x"], D["box"], D.get("csid"))
    return dict(core=core, cls=cls, s_ref=s_ref, n=n, ideal=ideal, cells=cells, box_ref=box)


def averaged_cell(S, fr):
    H = fr["H"]; x = fr["x"][S["core"]] - fr["lo"]
    s = np.linalg.solve(H, x.T).T
    ds = s - S["s_ref"]; ds -= np.rint(ds)                                # displacement, fractional of supercell
    u = ds * S["n"]                                                       # fractional of the small cell
    mean = np.array([u[S["cls"] == c].mean(0) for c in range(5)])
    mean -= mean[0]                                                       # Ba at its ideal site (origin choice)
    L = H / S["n"][None, :]                                               # small-cell vectors as columns
    pos = (S["ideal"] + mean) % 1.0
    return (L.T, pos, [56, 22, 8, 8, 8]), mean @ L.T                     # spglib cell; mean displacements (A)


def disp_phase(d, dmin=0.01, frac=0.3):
    """phase from the averaged Ti displacement relative to Ba (A): sorted |components| c1 >= c2 >= c3.
    C: c1 < dmin; T: c2 < frac*c1; O: c3 < frac*c2; else R."""
    c = np.sort(np.abs(d))[::-1]
    if c[0] < dmin:
        return "C"
    if c[1] < frac * c[0]:
        return "T"
    if c[2] < frac * c[1]:
        return "O"
    return "R"


def sg(cell, prec):
    try:
        d = spglib.get_symmetry_dataset(cell, symprec=prec)
        num = d.number if hasattr(d, "number") else d["number"]
        sym = d.international if hasattr(d, "international") else d["international"]
        return int(num), str(sym)
    except Exception:
        return 0, "none"


def frame_info(run):
    """step -> (T, stage, E) from traj_T.dat (ramps) or traj.dat (hysteresis); nominal ramp T if params.json."""
    info = {}
    if (run / "traj_T.dat").exists():
        for l in open(run / "traj_T.dat"):
            if not l.startswith("#"):
                p = l.split(); info[int(p[0])] = [float(p[1]), "ramp", 0.0]
        try:
            v = json.load(open(run / "params.json"))["lammps_vars"]
            T0, T1, nr, nh = (float(v[k]) for k in ("T0", "T1", "nramp", "nhold"))
            for st in info:
                info[st][0] = T0 if st < nh else (T0 + (T1 - T0) * (st - nh) / nr if st < nh + nr else T1)
        except (FileNotFoundError, KeyError):
            pass
    elif (run / "traj.dat").exists():
        tr = read_traj(run / "traj.dat")
        for st, T, s, E in zip(tr["step"].astype(int), tr["T"], tr["stage"], tr["E"]):
            info[int(st)] = [float(T), str(s), float(E)]
    return info


def windows(T, ph):
    """contiguous runs of one label that persist >= a.persist frames: (label, T_start, T_end, n)."""
    out, i = [], 0
    while i < len(ph):
        j = i
        while j + 1 < len(ph) and ph[j + 1] == ph[i]:
            j += 1
        if j - i + 1 >= a.persist:
            out.append((ph[i], T[i], T[j], j - i + 1))
        i = j + 1
    return out


summary = {}
for run in map(Path, a.runs):
    S = setup(run); info = frame_info(run)
    dump = run / "dump_T.lammpstrj" if (run / "dump_T.lammpstrj").exists() else run / "dump.lammpstrj"
    rows = []
    for k, fr in enumerate(iter_dump(dump)):
        if k % a.stride:
            continue
        cell, disp = averaged_cell(S, fr)
        sgs = [sg(cell, p) for p in a.symprec]
        pc = cell_polarization(S["cells"], fr["x"], fr["H"])
        net = np.linalg.norm(pc.mean(0)); loc = np.linalg.norm(pc, axis=1).mean()
        lf = local_fractions(pc)
        L = cell[0]; lens = np.linalg.norm(L, axis=1)
        ang = [np.degrees(np.arccos(L[i] @ L[j] / lens[i] / lens[j])) for i, j in ((1, 2), (0, 2), (0, 1))]
        T, stage, E = info.get(fr["step"], [np.nan, "?", np.nan])
        rows.append(dict(step=fr["step"], T=T, stage=stage, E=E,
                         **{f"sg_{p:g}": n for p, (n, _) in zip(a.symprec, sgs)},
                         **{f"sym_{p:g}": s for p, (_, s) in zip(a.symprec, sgs)},
                         netP=float(net), absP=float(loc), order=float(net / loc) if loc else 0.0,
                         a=float(lens[0]), b=float(lens[1]), c=float(lens[2]), alpha=ang[0], beta=ang[1], gamma=ang[2],
                         disp_phase=disp_phase(disp[1]), Ti_disp=disp[1].round(4).tolist(), **lf))
    out = run / "analysis"; out.mkdir(exist_ok=True)
    keys = [k for k in rows[0] if k != "Ti_disp"]
    with open(out / "phase_id.csv", "w") as f:
        f.write(",".join(keys + ["Ti_dx", "Ti_dy", "Ti_dz"]) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "," + ",".join(f"{v:.4f}" for v in r["Ti_disp"]) + "\n")
    pm = a.symprec[len(a.symprec) // 2]
    T = np.array([r["T"] for r in rows])
    lab = [f"{r[f'sym_{pm:g}']}({r[f'sg_{pm:g}']})" for r in rows]
    dl = [r["disp_phase"] for r in rows]
    print(f"\n== {run.name}: {len(rows)} frames, n = {S['n'].tolist()}, space groups at symprec {pm} A")
    for p in a.symprec:
        nums, cnt = np.unique([r[f"sg_{p:g}"] for r in rows], return_counts=True)
        print(f"   symprec {p:<6g}: " + ", ".join(f"{n}:{c}" for n, c in zip(nums, cnt)))
    w = windows(T, lab)
    for l, t0, t1, nfr in w:
        print(f"   {l:14s} {t0:7.1f} -> {t1:7.1f} K  ({nfr} frames)")
    print("   from the averaged Ti displacement (C/T/O/R):")
    wd = windows(T, dl)
    for l, t0, t1, nfr in wd:
        print(f"   {l:14s} {t0:7.1f} -> {t1:7.1f} K  ({nfr} frames)")
    low = [r for r in rows if r["order"] < 0.5]
    if low:
        print(f"   {len(low)} frames with |<P>|/<|P|> < 0.5 (multidomain or disordered): averaged-cell symmetry is an upper bound there")
    if info and any(r["stage"] not in ("ramp", "?") for r in rows):
        print("   by stage (LAST frame of each stage, e.g. 'eq' = equilibrated at T before any field):")
        last = {}
        for r, l in zip(rows, lab):
            last[r["stage"]] = (r, l)
        for r, l in last.values():
            if True:
                print(f"     {r['stage']:6s} E={r['E']:+.4f}  {l:14s} |<P>|={r['netP']:.3f} <|P|>={r['absP']:.3f} "
                      f"a,b,c={r['a']:.4f},{r['b']:.4f},{r['c']:.4f}")
    summary[run.name] = dict(windows=[dict(phase=l, T0=float(t0), T1=float(t1), frames=int(nf)) for l, t0, t1, nf in w],
                             disp_windows=[dict(phase=l, T0=float(t0), T1=float(t1), frames=int(nf)) for l, t0, t1, nf in wd],
                             symprec_summary=pm)
(ROOT / "figures" / "out").mkdir(parents=True, exist_ok=True)
json.dump(summary, open(ROOT / "figures" / "out" / "phase_id.json", "w"), indent=1)
print("\nwrote <run>/analysis/phase_id.csv and figures/out/phase_id.json")
