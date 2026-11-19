#!/usr/bin/env python3
"""Run process_run.py on every finished run (STATUS == complete) that has no analysis/metrics.json yet,
then write runs/summary.csv with one line of loop metrics per run."""
import csv, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader(l for l in open(ROOT / "tools/runs.csv") if not l.startswith("#")))
out = []
for r in rows:
    d = ROOT / "runs" / r["group"] / f"{r['T']}K" / r["run_id"]
    if not (d / "STATUS").exists() or (d / "STATUS").read_text().strip() != "complete":
        continue
    if not (d / "analysis" / "metrics.json").exists():
        rc = subprocess.run([sys.executable, str(ROOT / "analysis/process_run.py"), str(d)]).returncode
        if rc != 0:
            print(f"!! {r['run_id']}: analysis failed (exit {rc}) -- skipped, send the traceback"); continue
    m = json.load(open(d / "analysis" / "metrics.json"))
    g = m.get("loop_cell_all", m["global_loop"])
    out.append(dict(run_id=r["run_id"], T=r["T"], n=" ".join(f"{v:.3f}" for v in m["n"]),
                    Ec_V_per_A=round(g["Ec"], 4), Pr_C_m2=round(g["Pr"], 4), Ps_C_m2=round(g["Ps"], 4),
                    squareness=round(g["squareness"], 3), imprint=round(g["Ec_imprint"], 4),
                    P_offset=round(g["P_offset"], 4), multistep=g["n_steps_down"] + g["n_steps_up"] > 2))
if out:
    with open(ROOT / "runs/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"wrote runs/summary.csv ({len(out)} runs)")
else:
    print("no finished runs yet")
