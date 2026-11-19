"""Readers for BTO-GB-v2 run outputs (traj.dat, dump.lammpstrj, LAMMPS data files)."""
import json
from pathlib import Path
import numpy as np

TRAJ_COLS = ["step", "time_ps", "T", "E", "Ex", "Ey", "Ez", "lx", "ly", "lz",
             "pxx", "pyy", "pzz", "Px", "Py", "Pz", "Ppar"]


def read_traj(path):
    """traj.dat -> dict: 'stage' (str array) + one float array per column."""
    stages, vals = [], []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        stages.append(p[0])
        vals.append([float(x) for x in p[1:]])
    a = np.array(vals)
    out = {c: a[:, i] for i, c in enumerate(TRAJ_COLS)}
    out["stage"] = np.array(stages)
    return out


def read_data(path):
    """Minimal LAMMPS data reader (atom_style full) -> dict with box, id, type, q, x (sorted by id), and
    core->shell map from CS-Info (or from bonds if CS-Info absent)."""
    lines = Path(path).read_text().split("\n")
    box = np.zeros(3)
    for l in lines[:40]:
        for ax, key in enumerate(("xlo xhi", "ylo yhi", "zlo zhi")):
            if key in l:
                lo, hi = map(float, l.split()[:2]); box[ax] = hi - lo
    def section(name):
        for i, l in enumerate(lines):
            if l.strip().split("#")[0].strip() == name:
                j = i + 2; rows = []
                while j < len(lines) and lines[j].strip():
                    rows.append(lines[j].split()); j += 1
                return rows
        return []
    atoms = np.array([[float(v) for v in r[:7]] for r in section("Atoms")])
    order = np.argsort(atoms[:, 0])
    atoms = atoms[order]
    d = dict(box=box, id=atoms[:, 0].astype(int), type=atoms[:, 2].astype(int), q=atoms[:, 3], x=atoms[:, 4:7])
    cs = section("CS-Info")
    if cs:
        csid = {int(r[0]): int(r[1]) for r in cs}
        d["csid"] = np.array([csid[i] for i in d["id"]])
    meta = Path(str(path) + ".json")
    d["meta"] = json.load(open(meta)) if meta.exists() else {}
    return d


def iter_dump(path):
    """Yield frames from a 'custom id type xu yu zu' dump sorted by id (orthogonal or triclinic):
    dict(step, box (3,) edge lengths, H (3,3) cell matrix with cell vectors as columns, lo, id, type, x (N,3))."""
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(f.readline())
            f.readline(); n = int(f.readline())
            hdr = f.readline()
            b = [list(map(float, f.readline().split())) for _ in range(3)]
            if "xy" in hdr:   # triclinic: bounds are bounding-box values, convert to lo/hi + tilts
                xy, xz, yz = b[0][2], b[1][2], b[2][2]
                xlo = b[0][0] - min(0.0, xy, xz, xy + xz); xhi = b[0][1] - max(0.0, xy, xz, xy + xz)
                ylo = b[1][0] - min(0.0, yz);              yhi = b[1][1] - max(0.0, yz)
                zlo, zhi = b[2][0], b[2][1]
            else:
                xy = xz = yz = 0.0
                xlo, xhi, ylo, yhi, zlo, zhi = b[0][0], b[0][1], b[1][0], b[1][1], b[2][0], b[2][1]
            lo = np.array([xlo, ylo, zlo]); hi = np.array([xhi, yhi, zhi])
            H = np.array([[xhi - xlo, xy, xz], [0.0, yhi - ylo, yz], [0.0, 0.0, zhi - zlo]])  # columns = cell vectors
            cols = f.readline().split()[2:]
            arr = np.loadtxt([f.readline() for _ in range(n)])
            ic = {c: i for i, c in enumerate(cols)}
            yield dict(step=step, lo=lo, box=hi - lo, H=H, id=arr[:, ic["id"]].astype(int),
                       type=arr[:, ic["type"]].astype(int),
                       x=arr[:, [ic["xu"], ic["yu"], ic["zu"]]])


def load_params(run_dir):
    p = Path(run_dir) / "params.json"
    return json.load(open(p)) if p.exists() else {}
