#!/usr/bin/env python3
"""
Ferroelectric domain-wall energies in the R phase (0 K), referee point on domain walls.

  build   : runs/07_domain_walls/<wall>/{walls,mono}/ for wall in 109 (on {100}), 71 (on {110}),
            180 (on {110}); 'walls' = two R domains (P1 | P2, two periodic walls), 'mono' = P1
            everywhere in the same box (reference). One Slurm job relaxes all of them
            (templates/dw_energy.in: damped quench, triclinic 'tri' barostat).
  collect : sigma_DW = (E_walls - E_mono) / (2 A), A = |a x c| of the relaxed walls cell; checks that
            both domains survived the quench with the intended polarization directions.

  python tools/dw_scan.py build [--walltime 02:00:00]
  sbatch runs/07_domain_walls/job_dw.sh
  python tools/dw_scan.py collect
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "structures" / "gen_bto.py"
TPL = ROOT / "templates" / "dw_energy.in"
FF = ROOT / "forcefield" / "bto_coreshell.ff"
OUT = ROOT / "runs" / "07_domain_walls"
EV_A2_TO_MJ_M2 = 16021.8
SIZES = {"109": (6, 20, 6), "71": (4, 14, 6), "180": (4, 14, 6)}   # ~24 x 80 x 24 A, walls 40 A apart


def build(args):
    jobs = []
    todo = [(w + args.suffix, w, n) for w, n in SIZES.items()]
    # spacing-convergence variants: same wall, wall separation doubled (folder <wall>_2x<suffix>)
    todo += [(w + "_2x" + args.suffix, w, (SIZES[w][0], 2 * SIZES[w][1], SIZES[w][2])) for w in args.double]
    for name, wall, n in todo:
        for kind in ("walls", "mono"):
            d = OUT / name / kind; d.mkdir(parents=True, exist_ok=True)
            if (d / "DONE").exists():
                jobs.append(d); continue                      # never rebuild a finished relaxation
            cmd = [sys.executable, str(GEN), "domain", "--wall", wall, "--n", *map(str, n), "--a", "4.01",
                   "-o", str(d / "input.data")] + (["--mono"] if kind == "mono" else [])
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            (d / "in.lammps").write_text(TPL.read_text()); (d / "bto_coreshell.ff").write_text(FF.read_text())
            jobs.append(d)
        print(f"built {name}: n={n}")
    lines = [l.replace("{qos}", args.qos).replace("{nodes}", "1").replace("{ntasks}", "32")
             .replace("{walltime}", args.walltime).replace("{run_id}", "dw_scan")
             for l in (ROOT / "tools" / "submit_perlmutter.sh").read_text().split("\n")
             if not l.startswith("srun") and "RUN_COMPLETE" not in l and "STATUS" not in l
             and not l.startswith("echo")]
    body = [f"cd {OUT}"]
    for j in jobs:
        rel = j.relative_to(OUT)
        body.append(f"( cd {rel} && [ -f DONE ] || {{ srun -n 32 --cpu-bind=cores \"$LMP\" -in in.lammps "
                    f"-log log.lammps -var data input.data -var ff bto_coreshell.ff -var nb {args.nb} > /dev/null && touch DONE; }} )")
    (OUT / "job_dw.sh").write_text("\n".join(lines).replace("{{", "{").replace("}}", "}") + "\n" + "\n".join(body) + "\n")
    print(f"wrote {OUT/'job_dw.sh'} ({len(jobs)} relaxations, 32 ranks each; re-submitting skips finished ones)")


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
        if l.startswith("DWE_RESULT"):
            v = [float(x) for x in l.split()[1:]]
            r = dict(zip(("pe", "lx", "ly", "lz", "xy", "xz", "yz", "press", "fmax"), v))
            r["pe_relax"] = r["pe"]; r["shifted"] = True   # dw_energy.in was introduced with the shifted force field
            if e_eval is not None:
                r["pe"] = e_eval
            return r
    return None


def domain_P(d):
    """mean per-cell P in the interiors of the two domains (y_ref/Ly in 0.1-0.4 and 0.6-0.9)."""
    sys.path.insert(0, str(ROOT / "analysis"))
    from btoan.io import read_data, iter_dump
    from btoan.polarization import build_cells, cell_polarization
    Din = read_data(d / "input.data")
    cells = build_cells(Din["type"], Din["q"], Din["x"], Din["box"], Din.get("csid"))
    fr = next(iter_dump(d / "relaxed.dump"))
    pc = cell_polarization(cells, fr["x"], fr["H"])
    y = cells["ti_ref"][:, 1] / Din["box"][1]
    ok = ~cells["irregular"] if "irregular" in cells else np.ones(len(y), bool)
    m1 = ok & (y > 0.1) & (y < 0.4); m2 = ok & (y > 0.6) & (y < 0.9)
    return pc[m1].mean(0), pc[m2].mean(0), fr["H"], json.load(open(d / "input.data.json"))


PERIOD = {"109": 1.0, "71": 2 ** 0.5, "180": 2 ** 0.5}   # lab-y period of the cell, in units of a


def read_peatom(d):
    """per-atom energies: eval_peatom.dump (tools/dw_scan.py profile) or relaxed.dump if it has c_pa."""
    for f in (d / "eval_peatom.dump", d / "relaxed.dump"):
        if not f.exists():
            continue
        L = f.read_text().split("\n")
        i = next(k for k, l in enumerate(L) if l.startswith("ITEM: ATOMS"))
        cols = L[i].split()[2:]
        if "c_pa" not in cols:
            continue
        hdr = L[i - 4]; b = [list(map(float, L[k].split())) for k in range(i - 3, i)]
        if "xy" in hdr:
            xy, xz, yz = b[0][2], b[1][2], b[2][2]
            xlo = b[0][0] - min(0.0, xy, xz, xy + xz); xhi = b[0][1] - max(0.0, xy, xz, xy + xz)
            ylo = b[1][0] - min(0.0, yz); yhi = b[1][1] - max(0.0, yz)
        else:
            xy = xz = yz = 0.0; xlo, xhi, ylo, yhi = b[0][0], b[0][1], b[1][0], b[1][1]
        zlo, zhi = b[2][0], b[2][1]
        H = np.array([[xhi - xlo, xy, xz], [0, yhi - ylo, yz], [0, 0, zhi - zlo]])
        a = np.loadtxt([l for l in L[i + 1:] if l.strip()])
        c = {k: j for j, k in enumerate(cols)}
        x = a[:, [c["xu"], c["yu"], c["zu"]]] - np.array([xlo, ylo, zlo])
        return dict(id=a[:, c["id"]].astype(int), type=a[:, c["type"]].astype(int), x=x, pe=a[:, c["c_pa"]], H=H)
    return None


def energy_profile(dw, dm, meta):
    """excess energy per formula unit, one bin per lab-y period (offset 1/8 period so no layer is cut),
    relative to the single-domain reference. Walls sit between bins nb-1|0 and nb/2-1|nb/2.
    Returns (profile meV/f.u., sigma_local mJ/m2, interior mean meV/f.u., interior ramp meV/f.u.)"""
    W, M = read_peatom(dw), read_peatom(dm)
    if W is None or M is None:
        return None
    e_mono = M["pe"].sum() / (M["type"] == 2).sum()
    nb = int(meta["n"][1])
    s = np.linalg.solve(W["H"], W["x"].T).T[:, 1]
    ba = np.where(W["type"] == 1)[0]
    s0 = s[ba[np.argmin(np.abs(((s[ba] + 0.5) % 1.0) - 0.5))]]          # a Ba core next to the y=0 wall
    k = np.floor((((s - s0) % 1.0) * nb + 0.125)).astype(int) % nb
    nfu = np.bincount(k[W["type"] == 2], minlength=nb)
    E = np.bincount(k, weights=W["pe"], minlength=nb)
    prof = (E / nfu - e_mono) * 1000
    w = max(2, nb // 8)
    inner = np.array([(j % (nb // 2)) >= w and (j % (nb // 2)) < nb // 2 - w for j in range(nb)])
    d_int = prof[inner].mean()
    A = np.linalg.norm(np.cross(W["H"][:, 0], W["H"][:, 2]))
    sig_loc = ((prof - d_int) * nfu).sum() / 1000 / (2 * A) * EV_A2_TO_MJ_M2
    return prof, sig_loc, d_int, np.ptp(prof[inner][: inner.sum() // 2])


def profile(args):
    dirs = sorted(p.parent for p in OUT.glob("*/*/relaxed.data"))
    tpl = ROOT / "templates" / "energy_eval.in"
    lines = ["#!/bin/bash", "#SBATCH -A YOUR_NERSC_ACCOUNT", "#SBATCH -C cpu", "#SBATCH -q debug", "#SBATCH -N 1",
             "#SBATCH -t 00:20:00", "#SBATCH -J dwprof", f"#SBATCH -o {OUT}/slurm-dwprof-%j.out",
             "export OMP_NUM_THREADS=1", 'LMP=${LMP:-$HOME/bin/lmp}']
    for d in dirs:
        lines.append(f'( cd {d} && [ -f eval_peatom.dump ] || srun -n 16 --cpu-bind=cores "$LMP" -in {tpl} '
                     f'-log eval.log -var data relaxed.data -var ff {FF} -var peatom 1 > /dev/null )')
    (OUT / "job_dwprof.sh").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT/'job_dwprof.sh'} ({len(dirs)} structures)")


def collect(args):
    rows = []
    names = sorted((p.name for p in OUT.iterdir() if (p / "walls").is_dir() and (p / "mono").is_dir()),
                   key=lambda x: (x.split("_")[0], len(x), x))
    for wall in names:
        rw, rm = result(OUT / wall / "walls"), result(OUT / wall / "mono")
        if rw is None or rm is None:
            print(f"{wall}: not finished"); continue
        if rw["shifted"] != rm["shifted"]:
            print(f"{wall}: walls/mono energies from different sources (run tools/reeval.py)"); continue
        P1, P2, H, meta = domain_P(OUT / wall / "walls")
        Q1, Q2, _, _ = domain_P(OUT / wall / "mono")
        A = np.linalg.norm(np.cross(H[:, 0], H[:, 2]))
        sig = (rw["pe"] - rm["pe"]) / (2 * A) * EV_A2_TO_MJ_M2
        ang = lambda u, v: float(np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))
        e1, e2 = np.array(meta["P1_lab"]), np.array(meta["P2_lab"])
        r = dict(wall=wall, sigma_mJ_m2=round(sig, 1), area_A2=round(A, 2),
                 P_dom1=np.round(P1, 3).tolist(), P_dom2=np.round(P2, 3).tolist(),
                 angle_between_domains=round(ang(P1, P2), 1), dev_dom1=round(ang(P1, e1), 1),
                 dev_dom2=round(ang(P2, e2), 1), mono_absP=round(float(np.linalg.norm((Q1 + Q2) / 2)), 3),
                 mono_dev=round(ang((Q1 + Q2) / 2, e1), 1), press_walls=rw["press"], fmax_walls=rw["fmax"])
        r["energy_shifted"] = bool(rw["shifted"])
        ep = energy_profile(OUT / wall / "walls", OUT / wall / "mono", meta)
        if ep is not None:
            r.update(sigma_local_mJ_m2=round(ep[1], 1), interior_excess_meV_fu=round(ep[2], 2),
                     interior_ramp_meV_fu=round(ep[3], 2), profile_meV_fu=np.round(ep[0], 2).tolist())
        r["ok"] = r["dev_dom1"] < 15 and r["dev_dom2"] < 15 and r["mono_dev"] < 15
        rows.append(r)
    json.dump(rows, open(OUT / "dw_energies.json", "w"), indent=1)
    print("sigma = total (E_walls - E_mono)/2A;  sigma_loc = same minus the domain-interior excess (needs per-atom energies)")
    print(f"{'wall':>7s} {'sigma':>7s} {'sig_loc':>7s} {'int.exc':>7s} {'ramp':>6s} {'angle':>6s} {'P dom1':>24s} {'P dom2':>24s} {'dev1/dev2/mono':>15s}")
    for r in rows:
        loc = (f"{r['sigma_local_mJ_m2']:7.1f} {r['interior_excess_meV_fu']:7.2f} {r['interior_ramp_meV_fu']:6.2f}"
               if "sigma_local_mJ_m2" in r else f"{'-':>7s} {'-':>7s} {'-':>6s}")
        flag = "" if r["ok"] else "   <-- domains not as seeded: energy is NOT for this wall"
        print(f"{r['wall']:>7s} {r['sigma_mJ_m2']:7.1f} {loc} {r['angle_between_domains']:6.1f} {str(r['P_dom1']):>24s} "
              f"{str(r['P_dom2']):>24s} {r['dev_dom1']:5.1f}/{r['dev_dom2']:4.1f}/{r['mono_dev']:4.1f}{flag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd")
    sp.required = True  # (python 3.6 compatible)
    b = sp.add_parser("build"); b.add_argument("--walltime", default="02:00:00"); b.add_argument("--qos", default="premium")
    b.add_argument("--double", nargs="*", default=[], choices=sorted(SIZES),
                   help="also build <wall>_2x with the wall separation doubled (convergence check)")
    b.add_argument("--nb", type=int, default=8000, help="damped-NPT steps of the quench (default 8000)")
    b.add_argument("--suffix", default="", help="folder suffix, e.g. _long for a longer-relaxation set")
    sp.add_parser("collect")
    sp.add_parser("profile", help="write job_dwprof.sh: per-atom energies of every finished relaxation")
    a = ap.parse_args()
    {"build": build, "collect": collect, "profile": profile}[a.cmd](a)
