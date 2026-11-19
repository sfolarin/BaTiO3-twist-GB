#!/usr/bin/env python3
"""
Referee checks A1 + A3 at 0 K: is the pinned normal polarization of the Sigma5 bicrystal
  (i)  independent of the electrostatics (Wolf vs PPPM),
  (ii) independent of the starting state (relax from -P_y and from a +P_y seed),
  (iii) set by the termination sense (continuous TiO2|BaO vs reversed BaO|TiO2 -> sign flip)?
Each case is a damped-quench relaxation (templates/gb_energy.in) of the production Sigma5(210) cell.

  python tools/term_test.py build [--small]      # --small: 2x6 cell for a quick local test
  (cd runs/08_termination && sbatch job_term.sh)
  python tools/term_test.py collect
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "structures" / "gen_bto.py"
TPL = ROOT / "templates" / "gb_energy.in"
OUT = ROOT / "runs" / "08_termination"
FF = {"wolf": ROOT / "forcefield" / "bto_coreshell.ff", "pppm": ROOT / "forcefield" / "bto_coreshell_pppm.ff"}
# name: (stacking, seed (lab, A) or None, electrostatics)
CASES = {
    "cont_wolf":        ("continuous", None, "wolf"),
    "cont_wolf_seedP":  ("continuous", (0.0, 0.12, 0.0), "wolf"),
    "rev_wolf":         ("reversed", None, "wolf"),
    "cont_pppm":        ("continuous", None, "pppm"),
    "cont_pppm_seedP":  ("continuous", (0.0, 0.12, 0.0), "pppm"),
    "rev_pppm":         ("reversed", None, "pppm"),
}


def build(a):
    nrep, ncell = (2, 6) if a.small else (4, 10)
    lines = ["#!/bin/bash", "#SBATCH -A YOUR_NERSC_ACCOUNT", "#SBATCH -C cpu", f"#SBATCH -q {a.qos}", "#SBATCH -N 1",
             f"#SBATCH -t {a.walltime}", "#SBATCH -J term_test", "#SBATCH -o slurm-%j.out",
             "export OMP_NUM_THREADS=1", 'LMP=${LMP:-$HOME/bin/lmp}', f"cd {OUT}"]
    for name, (stack, seed, es) in CASES.items():
        d = OUT / name; d.mkdir(parents=True, exist_ok=True)
        base = [sys.executable, str(GEN), "twist", "--hk", "2", "1", "--nrep", str(nrep), "--ncell", str(ncell),
                "--a", "4.01", "--stacking", stack]
        subprocess.run(base + ["-o", str(d / "ref.data")], check=True, stdout=subprocess.DEVNULL)   # unseeded = P reference
        subprocess.run(base + (["--seed", *map(str, seed)] if seed else []) + ["-o", str(d / "input.data")],
                       check=True, stdout=subprocess.DEVNULL)
        (d / "in.lammps").write_text(TPL.read_text()); (d / "ff.in").write_text(FF[es].read_text())
        lines.append(f'( cd {name} && [ -f DONE ] || {{ srun -n 64 --cpu-bind=cores "$LMP" -in in.lammps -log log.lammps '
                     f'-var data input.data -var ff ff.in > /dev/null && touch DONE; }} )')
        print(f"built {name}: stacking={stack} seed={seed} electrostatics={es} cell={nrep}x{ncell}")
    (OUT / "job_term.sh").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT/'job_term.sh'}")


def collect(a):
    sys.path.insert(0, str(ROOT / "analysis"))
    from btoan.io import read_data
    from btoan.polarization import build_cells, cell_polarization
    from btoan.strain import grain_masks
    rows = {}
    print(f"{'case':17s} {'P grain1 (x,y,z)':>26s} {'P grain2 (x,y,z)':>26s} {'<|P|>':>6s} {'E (eV/f.u.)':>13s} "
          f"{'GB dipole mid/edge (e/A)':>25s}")
    for name in CASES:
        d = OUT / name
        if not (d / "relaxed.data").exists():
            print(f"{name:17s} not finished"); continue
        R = read_data(d / "ref.data"); D = read_data(d / "relaxed.data")
        cells = build_cells(R["type"], R["q"], R["x"], R["box"], R.get("csid"))
        pc = cell_polarization(cells, D["x"], D["box"])
        g1, g2 = grain_masks(cells["ti_ref"][:, 1] / R["box"][1], margin_frac=0.1)
        y = cells["ti_ref"][:, 1]; nb = int(round(R["box"][1] / 4.01)); k = np.clip((y / R["box"][1] * nb).astype(int), 0, nb - 1)
        prof = np.bincount(k, weights=pc[:, 1], minlength=nb) / np.maximum(np.bincount(k, minlength=nb), 1)
        pe = None
        for l in open(d / "log.lammps"):
            if l.startswith("GBE_RESULT"):
                pe = float(l.split()[1])
        nfu = int((R["type"] == 2).sum())
        # interface dipole: excess normal polarization of the two cell layers next to each GB relative to the
        # grain interiors, integrated over the layer thickness (per-cell P is origin independent)
        inner = np.r_[2:nb // 2 - 2, nb // 2 + 2:nb - 2]; p_in = prof[inner].mean()
        a_l = R["box"][1] / nb
        d_mid = (prof[nb // 2 - 1] + prof[nb // 2] - 2 * p_in) * a_l / 16.0218      # e/A  (C/m^2 * A / 16.02)
        d_edge = (prof[nb - 1] + prof[0] - 2 * p_in) * a_l / 16.0218
        rows_d = dict(dipole_GB_mid_e_per_A=round(float(d_mid), 4), dipole_GB_edge_e_per_A=round(float(d_edge), 4),
                      Py_interior=round(float(p_in), 3))
        rows[name] = dict(**rows_d, P_g1=pc[g1].mean(0).round(3).tolist(), P_g2=pc[g2].mean(0).round(3).tolist(),
                          absP=round(float(np.linalg.norm(pc[g1 | g2], axis=1).mean()), 3),
                          E_fu=pe / nfu if pe else None, Py_profile=prof.round(3).tolist())
        r = rows[name]
        print(f"{name:17s} {str(r['P_g1']):>26s} {str(r['P_g2']):>26s} {r['absP']:6.3f} "
              f"{r['E_fu'] if r['E_fu'] is None else round(r['E_fu'], 5):>13} "
              f"{r['dipole_GB_mid_e_per_A']:+12.4f}/{r['dipole_GB_edge_e_per_A']:+.4f}")
    json.dump(rows, open(OUT / "term_test.json", "w"), indent=1)
    if "cont_wolf" in rows and "cont_pppm" in rows:
        print(f"\nWolf vs PPPM, grain-interior P_y: {rows['cont_wolf']['P_g1'][1]:+.3f} / {rows['cont_pppm']['P_g1'][1]:+.3f}")
    print("layer P_y profiles (one value per cubic cell along y) in runs/08_termination/term_test.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd"); sp.required = True
    b = sp.add_parser("build"); b.add_argument("--small", action="store_true")
    b.add_argument("--qos", default="premium"); b.add_argument("--walltime", default="02:00:00")
    sp.add_parser("collect"); a = ap.parse_args()
    build(a) if a.cmd == "build" else collect(a)
