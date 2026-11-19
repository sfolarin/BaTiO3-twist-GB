#!/usr/bin/env python3
"""
Analyses requested by the review, from data that already exist (no new MD):

  inst     B1  instantaneous (unaveraged) per-cell P_z / P_y at the remanent state + per-cell time traces
               python analysis/referee_checks.py inst runs/04_sigma5/300K/s5_001_300K runs/04_sigma5/10K/s5_001_10K
  margins  B2  interior loop metrics for FIXED physical exclusion widths (4, 8, 12 A) instead of 5% of L_y
               python analysis/referee_checks.py margins runs/04_sigma5/10K/s5_001_10K runs/06_finite_size/10K/s5L_001_10K ...
  potential    planar-averaged charge -> macroscopic field/potential (diagnostic only: the bare field is dominated by
               small variations of P_y between layers; the interface dipole is reported by tools/term_test.py collect)
  csep     B5  core-shell separation (max, 99.9th percentile) along the field cycle, completed vs failed runs
               python analysis/referee_checks.py csep runs/04_sigma5/300K/s5_001_300K runs/04_sigma5/300K/s5_010_300K
All results: printed + figures/out/referee_<check>.png + figures/out/referee_<check>.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from btoan.io import read_data, iter_dump, read_traj          # noqa: E402
from btoan.loops import loop_metrics                           # noqa: E402

import matplotlib; matplotlib.use("Agg")                      # noqa: E402
import matplotlib.pyplot as plt                               # noqa: E402
OUT = ROOT / "figures" / "out"; OUT.mkdir(parents=True, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
E_PER_EA2_TO_V_PER_A = 180.95          # field (V/A) of a sheet of 1 e/A^2:  e/(eps0 A^2)


def pick(E):
    Em = np.nanmax(np.abs(E)); ip = int(np.argmax(E > 0.999 * Em)); im = int(np.argmax(E < -0.999 * Em))
    return ip + int(np.argmin(np.abs(E[ip:im]))), ip


def interior(z, width=None):
    """interior-cell mask: default = stored 5%-of-L_y grains; width (A) = fixed exclusion around y=0 and y=L/2."""
    if width is None:
        m = z["grain1"] | z["grain2"]
        return m if m.any() else ~z["gb"]
    y = z["ti_ref"][:, 1]; Ly = y.max() + 4.01 / 2
    if not z["grain2"].any():
        return np.ones(len(y), bool)                      # single crystal: no GB, every cell is interior
    d = np.minimum.reduce([np.abs(y), np.abs(y - Ly / 2), np.abs(y - Ly)])
    return d > width


# ----------------------------------------------------------------------------- B1
def inst(runs):
    fig, ax = plt.subplots(len(runs), 3, figsize=(15, 4.2 * len(runs)), squeeze=False)
    res = {}
    for r, run in enumerate(map(Path, runs)):
        z = np.load(run / "analysis" / "cellP.npz"); P, E = z["P"], z["E"]; m = interior(z)
        j, ip = pick(E); frames = range(max(0, j - 3), min(len(E), j + 4))
        rows = []
        for f in frames:
            pz = P[f][m, 2]; py = P[f][m, 1]
            rows.append(dict(frame=int(f), E=float(E[f]), mean_abs_Pz=float(np.abs(pz).mean()), abs_mean_Pz=float(abs(pz.mean())),
                             frac_absPz_gt_0p1=float((np.abs(pz) > 0.1).mean()), mean_Py=float(py.mean()),
                             mean_absP=float(np.linalg.norm(P[f][m], axis=1).mean())))
            ax[r, 0].hist(pz, bins=np.linspace(-0.35, 0.35, 57), histtype="step", lw=1.2)
        ax[r, 0].set(xlabel=r"instantaneous per-cell $P_z$ (C m$^{-2}$)", ylabel="cells", title=f"{run.name}: 7 frames around $P_r^+$")
        # time traces of 6 random interior cells over the whole cycle
        rng = np.random.default_rng(0); cells = rng.choice(np.where(m)[0], 6, replace=False)
        for k, c in enumerate(cells):
            ax[r, 1].plot(np.arange(len(E)), P[:, c, 2], lw=0.9, color=C[k % len(C)])
        ax[r, 1].plot(np.arange(len(E)), E / np.nanmax(np.abs(E)) * 0.3, color="k", lw=1.5, ls="--", label=r"$E/E_{\max}\times0.3$")
        ax[r, 1].axvline(j, color="0.5", lw=1); ax[r, 1].legend(fontsize=8)
        ax[r, 1].set(xlabel="frame (0.8 ps)", ylabel=r"per-cell $P_z$ (C m$^{-2}$)", title="six interior cells")
        a = np.abs(P[:, m, 2]).mean(1); b = np.abs(P[:, m, 2].mean(1))
        ax[r, 2].plot(E, a, color=C[0], label=r"$\langle|P_z|\rangle$"); ax[r, 2].plot(E, b, color=C[1], ls="--", label=r"$|\langle P_z\rangle|$")
        ax[r, 2].plot(E, np.abs(P[:, m, 1]).mean(1), color=C[2], ls=":", label=r"$\langle|P_y|\rangle$")
        ax[r, 2].set(xlabel=r"$E$ (V Å$^{-1}$)", ylabel=r"C m$^{-2}$", title="instantaneous averages"); ax[r, 2].legend(fontsize=8)
        res[run.name] = rows
        print(f"\n{run.name}  (instantaneous, interior cells)")
        print(f"{'frame':>5s} {'E':>7s} {'<|Pz|>':>7s} {'|<Pz>|':>7s} {'f(|Pz|>0.1)':>11s} {'<Py>':>7s} {'<|P|>':>6s}")
        for q in rows:
            print(f"{q['frame']:5d} {q['E']:7.4f} {q['mean_abs_Pz']:7.3f} {q['abs_mean_Pz']:7.3f} {q['frac_absPz_gt_0p1']:11.2f} "
                  f"{q['mean_Py']:7.3f} {q['mean_absP']:6.3f}")
    fig.tight_layout(); fig.savefig(OUT / "referee_inst.png", dpi=200)
    json.dump(res, open(OUT / "referee_inst.json", "w"), indent=1)
    print("wrote figures/out/referee_inst.png/.json")


# ----------------------------------------------------------------------------- B2
def margins(runs, widths=(4.0, 8.0, 12.0)):
    res = {}
    print(f"{'run':18s} {'excl. (A)':>9s} {'cells':>6s} {'Ec':>7s} {'Pr':>6s} {'Ps':>6s} {'offset':>7s} {'<Py>(Pr+)':>9s}")
    for run in map(Path, runs):
        z = np.load(run / "analysis" / "cellP.npz"); P, E, n = z["P"], z["E"], z["n"]
        res[run.name] = {}
        for w in (None,) + tuple(widths):
            m = interior(z, w)
            m_ = loop_metrics(E, (P[:, m] @ n).mean(1), np.nanmax(np.abs(E)))
            j, _ = pick(E); py = P[max(0, j - 3):j + 4][:, m, 1].mean()
            lab = "5% Ly" if w is None else f"{w:.0f}"
            res[run.name][lab] = dict(n_cells=int(m.sum()), Ec=m_["Ec"], Pr=m_["Pr"], Ps=m_["Ps"], offset=m_["P_offset"], Py_Pr=float(py))
            print(f"{run.name:18s} {lab:>9s} {m.sum():6d} {m_['Ec']:7.4f} {m_['Pr']:6.3f} {m_['Ps']:6.3f} {m_['P_offset']:+7.3f} {py:+9.3f}")
    json.dump(res, open(OUT / "referee_margins.json", "w"), indent=1)
    print("wrote figures/out/referee_margins.json")


# ----------------------------------------------------------------------------- B3
def _structure(folder):
    """positions + charges + box of a relaxed structure (relaxed.data / minimized.data) or of the time-averaged
    first cycle frame of a run (dump.lammpstrj), plus the as-built input for charges."""
    f = Path(folder)
    for name in ("relaxed.data", "minimized.data"):
        if (f / name).exists():
            D = read_data(f / name); return D["x"], D["q"], D["box"], D["type"]
    raise FileNotFoundError(f"no relaxed.data/minimized.data in {f}")


def potential(folders, dy=0.05, sigma=0.4):
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    res = {}
    for k, f in enumerate(map(Path, folders)):
        x, q, box, t = _structure(f)
        Lx, Ly, Lz = box; A = Lx * Lz
        y = np.mod(x[:, 1], Ly)
        nb = int(round(Ly / dy)); grid = (np.arange(nb) + 0.5) * Ly / nb
        rho = np.zeros(nb)                                        # e / A^3, Gaussian-smeared planar average
        kk = np.fft.rfftfreq(nb, d=Ly / nb) * 2 * np.pi
        hist = np.bincount(np.clip((y / Ly * nb).astype(int), 0, nb - 1), weights=q, minlength=nb) / (A * Ly / nb)
        rho = np.fft.irfft(np.fft.rfft(hist) * np.exp(-0.5 * (kk * sigma) ** 2), n=nb)
        # macroscopic average: two box filters of one layer period (a/2 for the (010) BaO/TiO2 stack) and a
        yb = np.sort(y[t == 1]); cut = np.where(np.diff(yb) > 0.5)[0]           # BaO layers -> layer period a
        centers = np.array([g.mean() for g in np.split(yb, cut + 1)])
        a = float(np.median(np.diff(centers)))                    # grain-interior layer period (GB gaps are outliers)
        sinc = lambda w: np.sinc(kk * w / (2 * np.pi))             # exact periodic box filter of width w (Fourier space)
        rho_m = np.fft.irfft(np.fft.rfft(rho) * sinc(a / 2) * sinc(a), n=nb)
        # Poisson in 1D with periodic, tin-foil boundary: E(y) = 180.95 * cumulative integral of rho, zero mean
        Ey = np.cumsum(rho_m) * (Ly / nb) * E_PER_EA2_TO_V_PER_A; Ey -= Ey.mean()
        phi = -np.cumsum(Ey) * (Ly / nb); phi -= phi.mean()
        g1 = (grid > 0.15 * Ly) & (grid < 0.35 * Ly); g2 = (grid > 0.65 * Ly) & (grid < 0.85 * Ly)
        res[f.name] = dict(E_grain1_V_per_A=float(Ey[g1].mean()), E_grain2_V_per_A=float(Ey[g2].mean()),
                           phi_step_V=float(phi.max() - phi.min()), Ly=float(Ly))
        ax[0].plot(grid, Ey, color=C[k % len(C)], label=f.name); ax[1].plot(grid, phi, color=C[k % len(C)])
        print(f"{f.name:20s} macroscopic E_y in grain 1 / grain 2: {Ey[g1].mean():+.4f} / {Ey[g2].mean():+.4f} V/A ; "
              f"potential range {phi.max() - phi.min():.2f} V")
    for x_ in ax:
        x_.axhline(0, color="0.7", lw=0.8)
    ax[0].set(ylabel=r"macroscopic $E_y$ (V Å$^{-1}$)"); ax[1].set(ylabel=r"$\bar\phi$ (V)", xlabel=r"$y$ (Å)")
    ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "referee_potential.png", dpi=200)
    json.dump(res, open(OUT / "referee_potential.json", "w"), indent=1)
    print("wrote figures/out/referee_potential.png/.json  (field of all charges, cores+shells, in the relaxed state)")


# ----------------------------------------------------------------------------- B5
def csep(runs):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5)); res = {}
    for k, run in enumerate(map(Path, runs)):
        tr = read_traj(run / "traj.dat"); s2E = dict(zip(tr["step"].astype(int), tr["E"])); s2st = dict(zip(tr["step"].astype(int), tr["stage"]))
        D = read_data(run / "input.data"); csid = D.get("csid")
        core = np.where(D["type"] <= 3)[0]; shell = np.where(D["type"] >= 4)[0]
        order = np.argsort(csid[shell]); sh_by_mol = shell[order]; co = core[np.argsort(csid[core])]
        Es, mx, p999, t_ = [], [], [], []
        for fr in iter_dump(run / "dump.lammpstrj"):
            if s2st.get(fr["step"]) != "cycle":
                continue
            d = fr["x"][sh_by_mol] - fr["x"][co]
            H = fr["H"]; s = np.linalg.solve(H, d.T).T; s -= np.round(s); d = (H @ s.T).T
            r = np.linalg.norm(d, axis=1)
            Es.append(s2E.get(fr["step"], np.nan)); mx.append(r.max()); p999.append(np.percentile(r, 99.9)); t_.append(fr["step"])
        Es, mx, p999 = map(np.array, (Es, mx, p999))
        res[run.name] = dict(max_sep_A=float(mx.max()) if len(mx) else None, p999_max_A=float(p999.max()) if len(p999) else None,
                             n_frames=len(mx), status=(run / "STATUS").read_text().strip() if (run / "STATUS").exists() else "?")
        ax.plot(np.arange(len(mx)), mx, color=C[k % len(C)], label=f"{run.name} (max)")
        ax.plot(np.arange(len(mx)), p999, color=C[k % len(C)], ls=":", lw=1)
        print(f"{run.name:18s} {res[run.name]['status']:12s} frames {len(mx):4d}  max core-shell separation {mx.max():.3f} A, "
              f"99.9th pct max {p999.max():.3f} A")
    ax.set(xlabel="cycle frame", ylabel="core–shell separation (Å)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "referee_csep.png", dpi=200)
    json.dump(res, open(OUT / "referee_csep.json", "w"), indent=1); print("wrote figures/out/referee_csep.png/.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("check", choices=["inst", "margins", "potential", "csep"])
    ap.add_argument("paths", nargs="+"); a = ap.parse_args()
    {"inst": inst, "margins": margins, "potential": potential, "csep": csep}[a.check](a.paths)
