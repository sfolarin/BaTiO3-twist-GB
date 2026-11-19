#!/usr/bin/env python3
"""
Single-point re-evaluation of relaxed structures with the CURRENT force field
(pair_modify shift yes; see forcefield/bto_coreshell.ff). Forces are unchanged by the shift, so
structures relaxed earlier stand; only their energies are recomputed.

  python tools/reeval.py runs/02_gb_energy runs/07_domain_walls     # writes runs/job_eval.sh
  sbatch runs/job_eval.sh                                            # debug queue, ~10-20 min
Each folder with relaxed.data gets eval.log containing "EVAL_RESULT pe"; gb_scan.py / dw_scan.py
collect use it in preference to the energy printed by the relaxation.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
dirs = []
for top in sys.argv[1:]:
    dirs += sorted(p.parent for p in Path(top).resolve().rglob("relaxed.data"))
if not dirs:
    sys.exit("no relaxed.data found under " + " ".join(sys.argv[1:]))
tpl = ROOT / "templates" / "energy_eval.in"; ff = ROOT / "forcefield" / "bto_coreshell.ff"
lines = ["#!/bin/bash", "#SBATCH -A YOUR_NERSC_ACCOUNT", "#SBATCH -C cpu", "#SBATCH -q debug", "#SBATCH -N 1",
         "#SBATCH -t 00:30:00", "#SBATCH -J reeval", "#SBATCH -o slurm-reeval-%j.out",
         "export OMP_NUM_THREADS=1", 'LMP=${LMP:-$HOME/bin/lmp}']
for d in dirs:
    lines.append(f'( cd {d} && grep -q EVAL_RESULT eval.log 2>/dev/null || srun -n 16 --cpu-bind=cores "$LMP" '
                 f'-in {tpl} -log eval.log -var data relaxed.data -var ff {ff} > /dev/null )')
lines.append('echo "reeval done: $(grep -l EVAL_RESULT $(find ' + " ".join(str(Path(t).resolve()) for t in sys.argv[1:])
             + ' -name eval.log) | wc -l) of ' + str(len(dirs)) + '"')
(ROOT / "runs" / "job_eval.sh").write_text("\n".join(lines) + "\n")
print(f"wrote runs/job_eval.sh for {len(dirs)} structures")
