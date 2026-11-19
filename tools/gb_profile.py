#!/usr/bin/env python3
"""
Referee round 2, major 2: separate the boundary energy from a bulk-state (grain-interior) contribution.

gamma = (E_bi - E_ref)/2A compares a bicrystal whose grain interiors have |P| = 0.26-0.27 C/m^2 with a reference
whose |P| = 0.24-0.25. If the interiors are in a different state, their energy difference per f.u. (Delta_int)
enters gamma multiplied by the grain thickness. From per-atom energies we get the excess energy per f.u. in each
(010) layer pair (one f.u. thick) relative to the reference cell's mean energy per f.u.:
    gamma_total = sum(prof * nfu) / 2A                   (= gb_scan gamma, consistency check)
    Delta_int   = excess per f.u. of the grain-interior layer pairs (> 1/4 grain from either boundary)
    gamma_loc   = sum((prof - Delta_int) * nfu) / 2A      (boundary energy with the interior state difference removed)
    bulk term   = gamma_total - gamma_loc = Delta_int * N_fu / 2A
Per-atom energies: LAMMPS compute pe/atom (pair and bond energies split equally between the two atoms).

  python tools/gb_profile.py build          # single points with per-atom energies for every boundary's best shift + ref
  sbatch runs/02_gb_energy/job_gbprof.sh     # debug queue
  python tools/gb_profile.py collect        # writes runs/02_gb_energy/gb_profiles.json and figures/out/tab_gb_loc.tex
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "02_gb_energy"
TPL = ROOT / "templates" / "energy_eval.in"
FF = ROOT / "forcefield" / "bto_coreshell.ff"
EV_A2_TO_J_M2 = 16.0218
sys.path.insert(0, str(ROOT / "tools"))
from dw_scan import read_peatom                     # noqa: E402  (per-atom dump reader, triclinic-safe)


def targets():
    rows = json.load(open(OUT / "gb_energies.json"))
    return [(r, OUT / f"S{r['hk'][0]}{r['hk'][1]}") for r in rows]


def build(_):
    lines = ["#!/bin/bash", "#SBATCH -A YOUR_NERSC_ACCOUNT", "#SBATCH -C cpu", "#SBATCH -q debug", "#SBATCH -N 1",
             "#SBATCH -t 00:30:00", "#SBATCH -J gbprof", "#SBATCH -o slurm-gbprof-%j.out",
             "export OMP_NUM_THREADS=1", 'LMP=${LMP:-$HOME/bin/lmp}']
    n = 0
    for r, d in targets():
        for sub in (d / "ref", d / r["best_shift"]):
            if not (sub / "relaxed.data").exists():
                print(f"missing {sub}/relaxed.data -- skipped"); continue
            lines.append(f'( cd {sub} && [ -f eval_peatom.dump ] || srun -n 32 --cpu-bind=cores "$LMP" -in {TPL} '
                         f'-log eval_pa.log -var data relaxed.data -var ff {FF} -var peatom 1 > /dev/null )')
            n += 1
    (OUT / "job_gbprof.sh").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT/'job_gbprof.sh'} ({n} single points)")


def profile(W, e_ref_fu):
    """excess energy per f.u. in (BaO, TiO2) layer pairs along y (boundaries at y = 0 and L/2).
    Layers are found by clustering the core y coordinates (gaps > 0.8 A), so no bin can cut a layer;
    each shell is assigned to the layer of its core (shell id = core id + 1). The sum over pairs is exact:
    sum(excess) = E_bi - N_fu * e_ref."""
    Ly = W["H"][1, 1]; y = W["x"][:, 1] % Ly; t = W["type"]
    core = np.where(t <= 3)[0]
    order = core[np.argsort(y[core])]; ys = y[order]
    gaps = np.diff(np.r_[ys, ys[0] + Ly])
    g0 = int(np.argmax(gaps))                                    # start just after the largest gap
    order = np.roll(order, -(g0 + 1)); ys = (y[order] - y[order[0]]) % Ly
    lid_sorted = np.r_[0, np.cumsum(np.diff(ys) > 0.8)]
    layer = np.full(len(t), -1); layer[order] = lid_sorted
    idpos = {i: k for k, i in enumerate(W["id"])}
    for k in np.where(t >= 4)[0]:
        layer[k] = layer[idpos[W["id"][k] - 1]]
    if (layer < 0).any():
        raise ValueError("unassigned particles")
    nl = layer.max() + 1
    E = np.bincount(layer, weights=W["pe"], minlength=nl)
    nti = np.bincount(layer[t == 2], minlength=nl); nba = np.bincount(layer[t == 1], minlength=nl)
    ymean = np.array([np.angle(np.exp(2j * np.pi * y[(layer == l) & (t <= 3)] / Ly).mean()) / (2 * np.pi) % 1.0 for l in range(nl)])
    i0 = int(np.argmax(nba > 0))                                 # pair layers as (BaO, TiO2)
    pairs = [np.roll(np.arange(nl), -i0)[2 * m:2 * m + 2] for m in range(nl // 2)]
    if nl % 2 or any(nba[p[0]] == 0 or nti[p[1]] == 0 for p in pairs):
        raise ValueError(f"layer pairing failed (nl={nl})")
    exc = np.array([E[p].sum() - nti[p].sum() * e_ref_fu for p in pairs])
    nfu = np.array([nti[p].sum() for p in pairs])
    yc = np.array([ymean[p].mean() for p in pairs])
    dist = np.minimum.reduce([yc, np.abs(yc - 0.5), 1 - yc])      # fractional distance to the nearer boundary
    inner = dist > 0.125
    order = np.argsort(yc)
    return exc[order], nfu[order], inner[order], yc[order]


def collect(_):
    res = []
    print(f"{'boundary':16s} {'gamma':>7s} {'gamma_chk':>9s} {'gamma_loc':>9s} {'D_int':>8s} {'bulk term':>9s}")
    print(f"{'':16s} {'J/m2':>7s} {'J/m2':>9s} {'J/m2':>9s} {'meV/fu':>8s} {'J/m2':>9s}")
    for r, d in targets():
        R, W = read_peatom(d / "ref"), read_peatom(d / r["best_shift"])
        if R is None or W is None:
            print(f"{r['boundary']:16s} per-atom energies missing"); continue
        e_ref = R["pe"].sum() / (R["type"] == 2).sum()
        try:
            exc, nfu, inner, yc = profile(W, e_ref)
        except ValueError as e:
            print(f"{r['boundary']:16s} {e}"); continue
        A = np.linalg.norm(np.cross(W["H"][:, 0], W["H"][:, 2]))
        g_tot = exc.sum() / (2 * A) * EV_A2_TO_J_M2
        d_int = exc[inner].sum() / nfu[inner].sum()                  # eV per f.u. in the grain interiors
        g_loc = (exc.sum() - d_int * nfu.sum()) / (2 * A) * EV_A2_TO_J_M2
        prof = exc / nfu
        res.append(dict(boundary=r["boundary"], theta=r["theta_total_deg"], gamma=r["gamma_J_m2"], gamma_check=round(g_tot, 4),
                        gamma_loc=round(g_loc, 4), D_int_meV_fu=round(d_int * 1000, 3), bulk_term=round(g_tot - g_loc, 4),
                        profile_meV_fu=(prof * 1000).round(2).tolist(), y_frac=yc.round(4).tolist()))
        q = res[-1]
        print(f"{q['boundary']:16s} {q['gamma']:7.3f} {q['gamma_check']:9.3f} {q['gamma_loc']:9.3f} {q['D_int_meV_fu']:8.2f} {q['bulk_term']:9.3f}")
    json.dump(res, open(OUT / "gb_profiles.json", "w"), indent=1)
    if res:
        g = np.array([q["gamma_loc"] for q in res]); b = np.array([q["bulk_term"] for q in res])
        print(f"\ngamma_loc range {g.min():.3f}-{g.max():.3f} J/m2; bulk term {b.min():+.3f} to {b.max():+.3f} J/m2")
        lines = [r"\begin{tabular}{lcccc}", r"\hline",
                 r"Boundary & $\gamma$ (J\,m$^{-2}$) & $\Delta_{\rm int}$ (meV/f.u.) & bulk term (J\,m$^{-2}$) & $\gamma_{\rm loc}$ (J\,m$^{-2}$) \\",
                 r"\hline"]
        for q in res:
            lines.append(fr"$\Sigma${q['boundary'][5:]} & {q['gamma']:.3f} & {q['D_int_meV_fu']:+.2f} & {q['bulk_term']:+.3f} & {q['gamma_loc']:.3f} \\")
        lines += [r"\hline", r"\end{tabular}"]
        (ROOT / "figures" / "out").mkdir(parents=True, exist_ok=True)
        (ROOT / "figures" / "out" / "tab_gb_loc.tex").write_text("\n".join(lines) + "\n")
        print("wrote runs/02_gb_energy/gb_profiles.json and figures/out/tab_gb_loc.tex")


if __name__ == "__main__":
    {"build": build, "collect": collect}[sys.argv[1] if len(sys.argv) > 1 else "x"](None) if len(sys.argv) > 1 and \
        sys.argv[1] in ("build", "collect") else sys.exit(__doc__)
