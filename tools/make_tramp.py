#!/usr/bin/env python3
"""
Build the two zero-field temperature-ramp runs for the phase sequence (referee R1):
  runs/01_benchmark/tramp_heat : 10 -> 600 K      runs/01_benchmark/tramp_cool : 600 -> 10 K
on a 12x12x12 single crystal (17,280 particles), dt 0.2 fs, 1.5M ramp steps (300 ps, ~2 K/ps).
  python tools/make_tramp.py [--qos regular]   then   (cd runs/01_benchmark/tramp_heat && sbatch job.sh)
Analysis: python analysis/tramp_analysis.py runs/01_benchmark/tramp_heat runs/01_benchmark/tramp_cool
"""
import argparse, json, shutil, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--qos", default="premium")
ap.add_argument("--ff", default="bto_coreshell.ff", help="force-field file in forcefield/ (e.g. bto_coreshell_A7200.ff)")
ap.add_argument("--tag", default="", help="suffix for the run folders, e.g. _A7200")
a = ap.parse_args()
struct = ROOT / "structures" / "data" / "sc_12.data"
if not struct.exists():
    raise SystemExit("missing structures/data/sc_12.data -- run tools/make_structures.sh")
for name, T0, T1 in (("tramp_heat" + a.tag, 10, 600), ("tramp_cool" + a.tag, 600, 10)):
    d = ROOT / "runs" / "01_benchmark" / name
    if (d / "traj_T.dat").exists():
        print(f"skip {name} (has output)"); continue
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "templates" / "tramp.in", d / "in.lammps")
    shutil.copy(ROOT / "forcefield" / a.ff, d / "bto_coreshell.ff")
    shutil.copy(struct, d / "input.data")
    v = dict(data="input.data", ff="bto_coreshell.ff", T0=T0, T1=T1, nramp=1500000, nhold=40000,
             dt=0.0002, nthermo=2000, ndump=20000)
    json.dump(dict(run_id=name, structure="sc_12", forcefield=a.ff, lammps_vars=v,
                   created=datetime.datetime.now().isoformat(timespec="seconds")), open(d / "params.json", "w"), indent=1)
    varstr = " ".join(f"-var {k} {val}" for k, val in v.items())
    job = (ROOT / "tools" / "submit_perlmutter.sh").read_text().format(
        run_id=name, nodes=1, ntasks=128, walltime="05:00:00", qos=a.qos, lmp_vars=varstr)
    (d / "job.sh").write_text(job); (d / "job.sh").chmod(0o755)
    print(f"built {name}: {T0} -> {T1} K")
