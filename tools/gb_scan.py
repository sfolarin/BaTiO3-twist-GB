#!/usr/bin/env python3
"""
Grain-boundary energy workflow (replaces GRAIN-BOUNDARY.py).

  build   : for each boundary, write a reference cell (no GB) and bicrystals for a grid of
            rigid in-plane translations of grain 2, plus one Slurm job that relaxes all of them.
  collect : gamma(shift) = (E_bi - E_ref) / (2 A_relaxed); report the minimum over shifts,
            the residual stress, and the polarization left in the grains (should be ~0: the
            energies are for the centrosymmetric reference state at 0 K -- say so in the paper).

  python tools/gb_scan.py build   --boundaries 2,1 3,1 5,1 4,1 3,2 5,3 4,3 7,1 6,1 5,4 6,5 8,1 7,5 8,3 --grid 4
  sbatch runs/02_gb_energy/job_gb.sh
  python tools/gb_scan.py collect
"""
import argparse, json, math, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "structures" / "gen_bto.py"
TPL = ROOT / "templates" / "gb_energy.in"
FF = ROOT / "forcefield" / "bto_coreshell.ff"
OUT = ROOT / "runs" / "02_gb_energy"
EV_A2_TO_J_M2 = 16.0218


def ncell_for(L, target_grain=40.0, a=4.01):
    return max(6, int(round(target_grain / a)))


def nrep_for(h, k, a=4.01, min_L=17.0):
    L1 = math.hypot(h, k) * a
    return max(1, int(math.ceil(min_L / L1)))   # in-plane period >= ~17 A (cutoff 11 A + margin)


def build(args):
    jobs = []
    for hk in args.boundaries:
        h, k = map(int, hk.split(","))
        nrep = nrep_for(h, k); ncell = args.ncell
        tag = f"S{h}{k}"
        d = OUT / tag; d.mkdir(parents=True, exist_ok=True)
        shifts = [(i / args.grid, j / args.grid) for i in range(args.grid) for j in range(args.grid)]
        sub = d / "ref"; sub.mkdir(exist_ok=True)
        subprocess.run([sys.executable, str(GEN), "twist", "--hk", str(h), str(k), "--nrep", str(nrep),
                        "--ncell", str(ncell), "--a", "4.01", "--reference", "-o", str(sub / "input.data")],
                       check=True, stdout=subprocess.DEVNULL)
        jobs.append(sub)
        for (sx, sz) in shifts:
            sub = d / f"shift_{sx:.3f}_{sz:.3f}"; sub.mkdir(exist_ok=True)
            subprocess.run([sys.executable, str(GEN), "twist", "--hk", str(h), str(k), "--nrep", str(nrep),
                            "--ncell", str(ncell), "--a", "4.01", "--shift", str(sx), str(sz),
                            "-o", str(sub / "input.data")], check=True, stdout=subprocess.DEVNULL)
            jobs.append(sub)
        print(f"built {tag}: nrep={nrep} ncell/grain={ncell} shifts={len(shifts)}")
    for j in jobs:
        (j / "in.lammps").write_text(TPL.read_text())
        (j / "bto_coreshell.ff").write_text(FF.read_text())
    lines = [l.replace("{qos}", args.qos).replace("{nodes}", "1").replace("{ntasks}", "128")
             .replace("{walltime}", args.walltime).replace("{run_id}", "gb_scan")
             for l in (ROOT / "tools" / "submit_perlmutter.sh").read_text().split("\n")
             if not l.startswith("srun") and "RUN_COMPLETE" not in l and "STATUS" not in l
             and not l.startswith("echo")]
    body = [f"cd {OUT}"]
    for j in jobs:
        rel = j.relative_to(OUT)
        body.append(f"( cd {rel} && [ -f DONE ] || {{ srun --cpu-bind=cores \"$LMP\" -in in.lammps -log log.lammps "
                    f"-var data input.data -var ff bto_coreshell.ff > /dev/null && touch DONE; }} )")
    (OUT / "job_gb.sh").write_text("\n".join(lines).replace("{{", "{").replace("}}", "}") + "\n" + "\n".join(body) + "\n")
    print(f"wrote {OUT/'job_gb.sh'}  ({len(jobs)} relaxations; re-submitting skips finished ones)")


def result(d):
    log = d / "log.lammps"
    if not log.exists():
        return None
    ev = d / "eval.log"      # single-point energy with the current force field (tools/reeval.py)
    e_eval = None
    if ev.exists():
        for l in open(ev):
            if l.startswith("EVAL_RESULT"):
                e_eval = float(l.split()[1])
    for l in open(log):
        if l.startswith("GBE_RESULT"):
            v = l.split()[1:]
            return dict(pe=e_eval if e_eval is not None else float(v[0]), pe_relax=float(v[0]), shifted=e_eval is not None,
                        lx=float(v[1]), ly=float(v[2]), lz=float(v[3]), press=float(v[4]), fmax=float(v[5]))
    return None


def collect(args):
    sys.path.insert(0, str(ROOT / "analysis"))
    from btoan.io import read_data
    from btoan.polarization import build_cells, cell_polarization
    rows = []
    for d in sorted(OUT.glob("S*")):
        ref = result(d / "ref")
        if ref is None:
            print(f"{d.name}: reference not finished"); continue
        meta = json.load(open(d / "ref" / "input.data.json"))
        best = None
        for s in sorted(d.glob("shift_*")):
            r = result(s)
            if r is None:
                continue
            if r["shifted"] != ref["shifted"]:
                print(f"{d.name}/{s.name}: energy source differs from reference (run tools/reeval.py) -- skipped")
                continue
            A = r["lx"] * r["lz"]
            g = (r["pe"] - ref["pe"]) / (2 * A) * EV_A2_TO_J_M2
            if best is None or g < best[0]:
                best = (g, s.name, r)
        if best is None:
            continue
        g, sname, r = best
        # polarization left in the GRAIN INTERIORS (GB cells excluded) of the relaxed reference
        # and of the lowest-energy bicrystal. gamma is only meaningful if both are in the same state.
        sys.path.insert(0, str(ROOT / "analysis"))
        from btoan.strain import grain_masks
        def interior_P(run):
            Din = read_data(run / "input.data"); D = read_data(run / "relaxed.data")
            cells = build_cells(Din["type"], Din["q"], Din["x"], Din["box"], Din.get("csid"))
            pc = cell_polarization(cells, D["x"], D["box"])
            g1, g2 = grain_masks(cells["ti_ref"][:, 1] / Din["box"][1], margin_frac=0.15)
            m = (g1 | g2) & ~cells["irregular"]
            return float(np.linalg.norm(pc[m], axis=1).mean())
        P_ref = interior_P(d / "ref"); P_bi = interior_P(d / sname)
        h, k = meta["hk"]
        rows.append(dict(boundary=f"Sigma{meta['sigma']}({h}{k}0)", hk=[h, k], sigma=meta["sigma"],
                         theta_total_deg=round(meta["theta_total_deg"], 2), gamma_J_m2=round(g, 4),
                         best_shift=sname, n_shifts=len(list(d.glob('shift_*'))), press_bar=r["press"],
                         fmax=r["fmax"], ref_press_bar=ref["press"],
                         grain_absP_ref=round(P_ref, 4), grain_absP_bicrystal=round(P_bi, 4),
                         state_mismatch=abs(P_ref - P_bi) > 0.03, energy_shifted=bool(ref["shifted"])))
    rows.sort(key=lambda x: x["theta_total_deg"])
    src = {r["energy_shifted"] for r in rows}
    print("energies:", "re-evaluated with pair_modify shift yes" if src == {True} else
          "AS RELAXED (no Born shift -- not reliable, run tools/reeval.py)" if src == {False} else "MIXED sources")
    json.dump(rows, open(OUT / "gb_energies.json", "w"), indent=1)
    print(f"{'boundary':16s} {'theta':>7s} {'gamma J/m2':>10s} {'best shift':>18s} {'|P| ref':>8s} {'|P| bi':>8s} {'press ref/bi (bar)':>20s}")
    for r in rows:
        flag = "  <-- polarization states differ: gamma not reliable" if r["state_mismatch"] else ""
        print(f"{r['boundary']:16s} {r['theta_total_deg']:7.2f} {r['gamma_J_m2']:10.4f} {r['best_shift']:>18s} "
              f"{r['grain_absP_ref']:8.3f} {r['grain_absP_bicrystal']:8.3f} {r['ref_press_bar']:9.0f}/{r['press_bar']:<9.0f}{flag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd")
    sp.required = True  # (python 3.6 compatible)
    b = sp.add_parser("build")
    b.add_argument("--boundaries", nargs="+", default=["2,1", "3,1", "5,1", "4,1", "3,2", "5,3", "4,3",
                                                       "7,1", "6,1", "5,4", "6,5", "8,1", "7,5", "8,3"])
    b.add_argument("--grid", type=int, default=4, help="grid x grid rigid shifts per boundary")
    b.add_argument("--ncell", type=int, default=10, help="cubic cells per grain along the GB normal")
    b.add_argument("--walltime", default="12:00:00")
    b.add_argument("--qos", default="premium")
    sp.add_parser("collect")
    a = ap.parse_args()
    build(a) if a.cmd == "build" else collect(a)
