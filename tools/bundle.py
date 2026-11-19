#!/usr/bin/env python3
"""
Pack several built hysteresis runs into ONE Slurm job (the premium QOS allows only 5 jobs per user,
running + pending). Each run gets its own nodes (nodes from its params.json) and all runs execute
concurrently as separate job steps:  srun -N n --exact ... &  ...  wait.
The job's node count is the sum and its walltime the longest of the runs.

  python tools/bundle.py NAME RUN_ID [RUN_ID ...]      -> runs/_bundles/NAME.sh
  then submit with tools/submit_free.sh (or: sbatch runs/_bundles/NAME.sh)

Runs that are complete, or already submitted (JOBID present without a failed STATUS), are skipped.
Each run still ends with its own STATUS file (complete / failed(code)), exactly as with its job.sh.
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "_bundles"


def find(rid):
    ds = [d for d in (ROOT / "runs").glob(f"*/*K/{rid}") if d.is_dir()]
    if len(ds) != 1:
        sys.exit(f"{rid}: expected one run folder, found {ds} (build it with tools/make_runs.py --only {rid})")
    return ds[0]


def secs(t):
    h, m, s = map(int, t.split(":")); return 3600 * h + 60 * m + s


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    name, ids = sys.argv[1], sys.argv[2:]
    steps, nodes, wall = [], 0, 0
    for rid in ids:
        d = find(rid)
        status = (d / "STATUS").read_text().strip() if (d / "STATUS").exists() else ""
        if status == "complete":
            print(f"skip   {rid} (complete)"); continue
        if (d / "JOBID").exists() and not status.startswith("failed"):
            print(f"skip   {rid} (already submitted, JOBID {(d / 'JOBID').read_text().strip()})"); continue
        p = json.load(open(d / "params.json"))
        m = re.search(r'^srun --cpu-bind=cores "\$LMP" (.*)$', (d / "job.sh").read_text(), re.M)
        if not m:
            sys.exit(f"{rid}: srun line not found in job.sh")
        n, ntask = int(p["nodes"]), int(p["ntasks_per_node"])
        nodes += n; wall = max(wall, secs(p["walltime"]))
        steps.append(
            f'( cd {d} && echo "$SLURM_JOB_ID" > JOBID && '
            f'srun -N {n} -n {n * ntask} --ntasks-per-node={ntask} -c 2 --exact --cpu-bind=cores "$LMP" {m.group(1)} '
            f'> srun.out 2>&1; s=$?; '
            f'if grep -q RUN_COMPLETE log.lammps; then echo complete > STATUS; else echo "failed($s)" > STATUS; fi; '
            f'echo "{rid}: $(cat STATUS) $(date)" ) &')
        (d / "BUNDLE").write_text(name + "\n")
        print(f"add    {rid:18s} {n} node(s), {p['walltime']}")
    if not steps:
        sys.exit("nothing to bundle")
    OUT.mkdir(parents=True, exist_ok=True)
    hh, mm = divmod(wall // 60, 60)
    hdr = ["#!/bin/bash", "#SBATCH -A YOUR_NERSC_ACCOUNT", "#SBATCH -C cpu", "#SBATCH -q premium", f"#SBATCH -N {nodes}",
           f"#SBATCH -t {hh:02d}:{mm:02d}:00", f"#SBATCH -J {name}", f"#SBATCH -o {OUT}/{name}-%j.out",
           "#SBATCH --mail-type=END,FAIL", "#SBATCH --mail-user=YOUR_EMAIL",
           "export OMP_NUM_THREADS=1 OMP_PLACES=cores OMP_PROC_BIND=spread",
           "LMP=${LMP:-/path/to/lmp}",
           f'echo "bundle {name}: {len(steps)} runs on $SLURM_JOB_NUM_NODES nodes, start $(date)"']
    (OUT / f"{name}.sh").write_text("\n".join(hdr + steps + ["wait", f'echo "bundle {name} end $(date)"']) + "\n")
    print(f"wrote {OUT / (name + '.sh')}: {len(steps)} runs, {nodes} nodes, {hh:02d}:{mm:02d}:00")


if __name__ == "__main__":
    main()
