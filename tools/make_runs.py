#!/usr/bin/env python3
"""
Build run folders from tools/runs.csv. Nothing in runs/ is written by hand.

  python tools/make_runs.py                 # build every run not yet built
  python tools/make_runs.py --only s5_      # build runs whose id starts with s5_
  python tools/make_runs.py --test          # tiny 'seg' for a quick debug-queue check
  python tools/make_runs.py --force ...     # rebuild (refuses if the run already has output)

Each run folder gets:
  in.lammps   frozen copy of templates/hysteresis.in
  params.json every parameter + SHA256 of template, force field and structure
  job.sh      Slurm script (Perlmutter CPU)
  input.data  copy of the structure file
"""
import argparse, csv, hashlib, json, os, shutil, subprocess, sys, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "hysteresis.in"
FF = ROOT / "forcefield" / "bto_coreshell.ff"
SLURM = ROOT / "tools" / "submit_perlmutter.sh"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def git_rev():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "no-git"


def rows():
    with open(ROOT / "tools" / "runs.csv") as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="prefix filter on run_id")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--test", action="store_true", help="seg=200 and 30-min debug queue")
    a = ap.parse_args()

    for r in rows():
        rid = r["run_id"]
        if not rid.startswith(a.only):
            continue
        d = ROOT / "runs" / r["group"] / f"{r['T']}K" / rid
        if a.test:
            d = ROOT / "runs" / "_tests" / rid
        if d.exists():
            if not a.force:
                print(f"skip   {rid} (exists)")
                continue
            if (d / "traj.dat").exists():
                sys.exit(f"REFUSING to overwrite {d}: it already has output. Move it aside first.")
            shutil.rmtree(d)
        struct = ROOT / "structures" / "data" / f"{r['structure']}.data"
        if not struct.exists():
            sys.exit(f"missing structure {struct} -- run tools/make_structures.sh first")
        d.mkdir(parents=True)
        shutil.copy(TEMPLATE, d / "in.lammps")
        shutil.copy(struct, d / "input.data")
        if Path(str(struct) + ".json").exists():
            shutil.copy(str(struct) + ".json", d / "input.data.json")

        shutil.copy(FF, d / "bto_coreshell.ff")   # frozen copy: the run never depends on later edits
        v = dict(data="input.data", T=r["T"], dx=r["dx"], dy=r["dy"], dz=r["dz"],
                 baro=r["baro"], ff="bto_coreshell.ff")
        # timestep rule: 0.4 fs at low T; 0.2 fs at T >= 200 K with seg/nthermo/ndump doubled so the
        # physical protocol (times, field rate, sampling) is identical. (s5_010_300K lost an O shell at the
        # GB with 0.4 fs, 2026-10-31.)
        if float(r["T"]) >= 200:
            v.update(dt="0.0002", seg="20000", nthermo="1000", ndump="4000")
        for kv in (r.get("extra_vars") or "").split(";"):     # optional last column, e.g. By=0.03;Emax=0.05
            if kv.strip():
                k, val = kv.split("="); v[k.strip()] = val.strip()
        if v["ff"] != "bto_coreshell.ff":                      # alternative force field, e.g. ff=bto_coreshell_pppm.ff
            alt = ROOT / "forcefield" / v["ff"]
            if not alt.exists():
                sys.exit(f"{rid}: force field {alt} not found")
            shutil.copy(alt, d / v["ff"])
        if a.test:
            v["seg"] = "200" if float(r["T"]) < 200 else "400"
        params = dict(**{k: r[k] for k in r}, lammps_vars=v,
                      created=datetime.datetime.now().isoformat(timespec="seconds"),
                      git=git_rev(), sha_template=sha(TEMPLATE), sha_forcefield=sha(FF),
                      sha_structure=sha(struct), test=a.test)
        json.dump(params, open(d / "params.json", "w"), indent=1)

        varstr = " ".join(f"-var {k} {val}" for k, val in v.items())
        job = SLURM.read_text().format(
            run_id=rid, nodes=1 if a.test else r["nodes"], ntasks=r["ntasks_per_node"],
            walltime="00:30:00" if a.test else r["walltime"], qos="debug" if a.test else r["qos"],
            lmp_vars=varstr)
        (d / "job.sh").write_text(job)
        os.chmod(d / "job.sh", 0o755)
        print(f"built  {rid:18s} -> {d.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
