#!/usr/bin/env python3
"""Print the state of every run in tools/runs.csv: not built / built / running / complete / failed.
   python tools/status.py
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader(l for l in open(ROOT / "tools/runs.csv") if not l.startswith("#")))
print(f"{'run_id':18s} {'state':10s} {'cycle pts':>9s}  job")
for r in rows:
    d = ROOT / "runs" / r["group"] / f"{r['T']}K" / r["run_id"]
    if not d.exists():
        state, n, job = "not built", 0, ""
    else:
        job = (d / "JOBID").read_text().strip() if (d / "JOBID").exists() else ""
        st = (d / "STATUS").read_text().strip() if (d / "STATUS").exists() else ""
        n = sum(1 for l in open(d / "traj.dat") if l.startswith("cycle")) if (d / "traj.dat").exists() else 0
        state = st or ("running" if job else "built")
    print(f"{r['run_id']:18s} {state:10s} {n:9d}  {job}")
