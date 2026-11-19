#!/usr/bin/env python3
"""
0 K phase energetics with the LAMMPS BINARY (no Python LAMMPS module needed), to compare Ti-O Buckingham
A = 7220.27 eV (used in this work) with the published scaled-model value A = 7200.27 eV (Vielma & Schneider 2013,
Table IV).

Each phase is a 2x2x2 supercell (40 ions) seeded along its polar axis and relaxed with the same damped quench as
the boundary-energy runs, under symmetry constraints imposed through the forces:
  C: ions fixed at ideal sites (fix setforce 0 0 0), box relaxed isotropically      (Pm-3m)
  T: ions move along z only (fix setforce 0 0 NULL), aniso barostat                  (P4mm, P || [001])
  O: ions move in the y-z plane (fix setforce 0 NULL NULL), triclinic barostat       (Amm2, P || [011])
  R: free, triclinic barostat                                                         (R3m,  P || [111])
No random velocities are assigned, so the seeded symmetry is kept exactly.

  python tools/phase_bench_lmp.py build              # writes runs/01_benchmark/lmpbench/<A>/<phase>/
  python tools/phase_bench_lmp.py run [--lmp PATH]   # runs the 8 small jobs serially (login node is fine)
  python tools/phase_bench_lmp.py collect            # table + runs/01_benchmark/lmpbench/phase_bench_lmp.json
"""
import argparse, json, os, re, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "01_benchmark" / "lmpbench"
sys.path.insert(0, str(ROOT / "structures"))
import gen_bto  # noqa: E402

A_VALUES = {"A7220": 7220.27, "A7200": 7200.27}
SEED = {"C": None, "T": (0, 0, 1), "O": (0, 1, 1), "R": (1, 1, 1)}
CONSTRAINT = {"C": "fix cs all setforce 0.0 0.0 0.0", "T": "fix cs all setforce 0.0 0.0 NULL",
              "O": "fix cs all setforce 0.0 NULL NULL", "R": ""}
BARO = {"C": "iso 0.0 0.0 1.0", "T": "aniso 0.0 0.0 1.0", "O": "tri 0.0 0.0 1.0", "R": "tri 0.0 0.0 1.0"}
N, A0, D = 2, 4.00, 0.12

TEMPLATE = """units           metal
dimension       3
boundary        p p p
atom_style      full
timestep        0.0004
fix             csinfo all property/atom i_CSID
read_data       input.data fix csinfo NULL CS-Info
include         ff.in
group           cores  type 1 2 3
group           shells type 4 5 6
compute         CSequ all temp/cs cores shells
thermo          1000
thermo_style    custom step pe fmax press lx ly lz xy xz yz
{constraint}
fix             q  all nve/limit 0.02
fix             vq all viscous 0.5
run             3000
unfix           q
unfix           vq
fix             qb all npt temp 1.0 1.0 0.1 {baro}
fix_modify      qb temp CSequ
fix             vq all viscous 0.2
run             12000
unfix           qb
unfix           vq
run             0
write_data      relaxed.data
print           "BENCH_RESULT $(pe:%.8f) $(lx:%.6f) $(ly:%.6f) $(lz:%.6f) $(xy:%.6f) $(xz:%.6f) $(yz:%.6f) $(press:%.2f) $(fmax:%.3e)"
"""


def build(_):
    ff0 = (ROOT / "forcefield" / "bto_coreshell.ff").read_text()
    m = re.search(r"pair_coeff\s+5 6\s+([0-9.]+)", ff0); assert m, "force field Ti-O line not found"
    tio_line = m.group(0)
    for tag, A in A_VALUES.items():
        for ph, seed in SEED.items():
            d = OUT / tag / ph; d.mkdir(parents=True, exist_ok=True)
            ions, box = gen_bto.single_crystal((N, N, N), A0)
            if seed is not None:
                n = np.array(seed, float); n /= np.linalg.norm(n)
                ions = [(t, *(np.array(r) + (D * n if t == 2 else -0.5 * D * n if t == 3 else 0.0)))
                        for t, *r in ions]
            gen_bto.write_data(str(d / "input.data"), ions, box, f"BTO {ph} 2x2x2 seed {seed}")
            txt = (d / "input.data").read_text()
            if ph in ("O", "R"):                          # triclinic box so the barostat can tilt it
                txt = txt.replace(" zlo zhi\n", " zlo zhi\n0.0 0.0 0.0 xy xz yz\n", 1)
            (d / "input.data").write_text(txt)
            (d / "ff.in").write_text(ff0.replace(tio_line, tio_line.replace(m.group(1), f"{A:.2f}")))
            (d / "in.lammps").write_text(TEMPLATE.format(constraint=CONSTRAINT[ph], baro=BARO[ph]))
    print(f"built {len(A_VALUES) * len(SEED)} cases under {OUT}")


def run(a):
    lmp = a.lmp or os.environ.get("LMP") or "/path/to/lmp"
    for tag in A_VALUES:
        for ph in SEED:
            d = OUT / tag / ph
            r = subprocess.run([lmp, "-in", "in.lammps", "-log", "log.lammps", "-screen", "none"], cwd=d,
                               env=dict(os.environ, MPICH_GPU_SUPPORT_ENABLED="0"))   # login node: no GPU MPI
            ok = "BENCH_RESULT" in (d / "log.lammps").read_text() if (d / "log.lammps").exists() else False
            print(f"{tag} {ph}: {'ok' if ok and r.returncode == 0 else 'FAILED (see log.lammps)'}", flush=True)


def lattice(lx, ly, lz, xy, xz, yz):
    H = np.array([[lx, xy, xz], [0, ly, yz], [0, 0, lz]]) / N
    L = np.linalg.norm(H, axis=0)
    ang = lambda i, j: float(np.degrees(np.arccos(H[:, i] @ H[:, j] / L[i] / L[j])))
    return L, (ang(1, 2), ang(0, 2), ang(0, 1))


def collect(_):
    res = {}
    for tag in A_VALUES:
        res[tag] = {}
        for ph in SEED:
            log = OUT / tag / ph / "log.lammps"
            line = next((l for l in open(log) if l.startswith("BENCH_RESULT")), None) if log.exists() else None
            if line is None:
                print(f"{tag} {ph}: no result"); continue
            v = list(map(float, line.split()[1:]))
            L, ang = lattice(*v[1:7])
            res[tag][ph] = dict(E_fu=v[0] / N ** 3, a=sorted(L.tolist()), angles=ang, press=v[7], fmax=v[8])
        if "C" in res[tag]:
            for ph in res[tag]:
                res[tag][ph]["dE_meV_fu"] = (res[tag][ph]["E_fu"] - res[tag]["C"]["E_fu"]) * 1000
    json.dump(res, open(OUT / "phase_bench_lmp.json", "w"), indent=1)
    print(f"\n{'phase':5s}" + "".join(f"{t + ' dE':>12s}{t + ' a,b,c (A)':>28s}" for t in A_VALUES) + f"{'diff dE':>10s}")
    for ph in SEED:
        row = f"{ph:5s}"
        for t in A_VALUES:
            r = res[t].get(ph)
            row += f"{r['dE_meV_fu']:12.2f}{' '.join(f'{x:.4f}' for x in r['a']):>28s}" if r and "dE_meV_fu" in r else f"{'-':>12s}{'-':>28s}"
        if all(ph in res[t] and "dE_meV_fu" in res[t][ph] for t in A_VALUES):
            row += f"{res['A7200'][ph]['dE_meV_fu'] - res['A7220'][ph]['dE_meV_fu']:10.2f}"
        print(row)
    old = ROOT / "runs" / "01_benchmark" / "phase_bench_N2_TiO7220_shift.json"
    if old.exists():
        o = {r["phase"]: r["dE_meV_fu"] for r in json.load(open(old))}
        print("\ncheck: A7220 with this method vs the earlier Python benchmark (meV/f.u.): "
              + ", ".join(f"{p} {res['A7220'][p]['dE_meV_fu']:.1f} vs {o[p]:.1f}" for p in SEED if p in res["A7220"] and p in o))
    print(f"\nwrote {OUT / 'phase_bench_lmp.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd"); sp.required = True
    sp.add_parser("build"); r = sp.add_parser("run"); r.add_argument("--lmp"); sp.add_parser("collect")
    a = ap.parse_args(); {"build": build, "run": run, "collect": collect}[a.cmd](a)
