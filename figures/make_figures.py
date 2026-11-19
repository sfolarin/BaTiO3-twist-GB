#!/usr/bin/env python3
"""
All manuscript + SI figures and tables from the processed runs. Reads only analysis outputs:
  runs/<...>/analysis/{frames.csv, metrics.json, cellP.npz}, runs/01_benchmark/*shift.json,
  runs/01_benchmark/tramp_{heat,cool}/analysis/tramp.csv, runs/02_gb_energy/gb_energies.json,
  runs/07_domain_walls/dw_energies.json, structures/data/s5_210.data
Writes figures/out/Fig{1..6}.{pdf,png}, FigS*.{pdf,png}, tab_*.tex and numbers.json (every value
quoted in the text, so the manuscript never carries a hand-copied number).

  python figures/make_figures.py            # everything that has data; missing inputs are reported, not fatal
  python figures/make_figures.py --only 5   # one figure
Conventions: field in V/A, P in C/m^2, strain in %; P_par = grain-interior per-cell P projected on the field
direction n (GB cells excluded); 'interior' = cells farther than 5% of L_y from either GB.
"""
import argparse, json, sys, traceback
from pathlib import Path
import numpy as np
import logging, matplotlib; matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from btoan.loops import loop_metrics                       # noqa: E402

OUT = ROOT / "figures" / "out"
RUNS = ROOT / "runs"
EMAX = 0.06
A_LAT = 4.01

# ============================================================================= #
# PUBLICATION PLOT STYLE (as specified)
# ============================================================================= #
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 13, "font.weight": "bold",
    "axes.labelsize": 17, "axes.labelweight": "bold", "axes.titlesize": 16, "axes.titleweight": "bold",
    "axes.linewidth": 1.4, "xtick.labelsize": 13, "ytick.labelsize": 13, "xtick.direction": "in",
    "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "xtick.major.size": 6,
    "ytick.major.size": 6, "xtick.major.width": 1.2, "ytick.major.width": 1.2, "xtick.minor.visible": True,
    "ytick.minor.visible": True, "xtick.minor.size": 3, "ytick.minor.size": 3, "legend.fontsize": 10,
    "legend.framealpha": 0.95, "pdf.fonttype": 42, "ps.fonttype": 42,
})
# categorical palette, fixed order (colour-vision-deficiency checked); always paired with a line style or marker
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GREY = "#52514e"
LS = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
MK = ["o", "s", "^", "D", "v", "P", "X", "*"]
NUM = {}                                                    # every number quoted in the text


# ----------------------------------------------------------------------------- helpers
def rd(group, T, rid):
    """run directory; falls back to any group that holds <T>K/<rid> (the size and control runs were filed by hand)."""
    d = RUNS / group / f"{T}K" / rid
    if not d.exists():
        hits = sorted(RUNS.glob(f"*/{T}K/{rid}"))
        if hits:
            return hits[0]
    return d


def have(*paths):
    miss = [str(p.relative_to(ROOT)) for p in paths if not Path(p).exists()]
    if miss:
        raise FileNotFoundError("missing: " + ", ".join(miss))


def frames(run):
    f = run / "analysis" / "frames.csv"; have(f)
    a = np.genfromtxt(f, delimiter=",", names=True)
    g2 = a["Pcell_par_g2"]
    interior = a["Pcell_par_g1"] if np.all(np.isnan(g2)) else 0.5 * (a["Pcell_par_g1"] + g2)
    return dict(E=a["E"], P=interior, Pall=a["Pcell_par_all"], s1=a["strain_par_g1"] * 100,
                s2=a["strain_par_g2"] * 100)


def metrics_interior(run):
    F = frames(run)
    return loop_metrics(F["E"], F["P"], EMAX)


def cellp(run):
    f = run / "analysis" / "cellP.npz"; have(f)
    z = np.load(f)
    inner = z["grain1"] | z["grain2"]
    if not inner.any():
        inner = ~z["gb"]
    return z, inner


def pick(E):
    """frame indices: first +Emax, E=0 after it (P_r+), first -Emax, E=0 after it (P_r-)."""
    Em = np.nanmax(np.abs(E))
    ip = int(np.argmax(E > 0.999 * Em)); im = int(np.argmax(E < -0.999 * Em))
    return dict(pmax=ip, rp=ip + int(np.argmin(np.abs(E[ip:im]))), mmax=im, rm=im + int(np.argmin(np.abs(E[im:]))))


def tavg(P, i, w=3, side="both"):
    """time average of per-cell P around frame i. side="before": frames i-w..i only (use at the remanent states,
    so the window stays on the branch that has not yet switched; a symmetric window crosses -Ec when Ec is small)."""
    if side == "before":
        return P[max(0, i - w):i + 1].mean(0)
    return P[max(0, i - w):i + w + 1].mean(0)


def trem(P, k, key):
    """remanent states (rp, rm): the instantaneous frame at E = 0 (no field-contaminated window; layer averages
    over ~10^2 cells keep the noise small); field extrema: 7-frame average."""
    return P[k[key]] if key in ("rp", "rm") else tavg(P, k[key])


def layer_profile(z, Pt, comp):
    """per-cell value comp (0,1,2) averaged in layers of one lattice constant along y."""
    y = z["ti_ref"][:, 1]; Ly = y.max() + A_LAT / 2
    nb = int(round(Ly / A_LAT)); k = np.clip((y / Ly * nb).astype(int), 0, nb - 1)
    v = np.bincount(k, weights=Pt[:, comp], minlength=nb) / np.maximum(np.bincount(k, minlength=nb), 1)
    return (np.arange(nb) + 0.5) * Ly / nb, v, Ly


def label(ax, s, x=None, y=None):
    """panel letter just above the top-left corner of the axes (never collides with data or legends)."""
    ax.text(0.0, 1.015, f"({s})", transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="left")


def axes0(ax):
    ax.axhline(0, color="0.75", lw=0.9, zorder=0); ax.axvline(0, color="0.75", lw=0.9, zorder=0)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); fig.savefig(OUT / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig); print(f"  wrote figures/out/{name}.pdf/.png")


EXP_T = {"R-O": 183, "O-T": 278, "T-C": 393}                 # BaTiO3 experiment (K)


# ============================================================================= Fig. 1: model validation
def fig1():
    bench = RUNS / "01_benchmark" / "phase_bench_N2_TiO7220_shift.json"
    heat = RUNS / "01_benchmark" / "tramp_heat" / "analysis" / "tramp.csv"
    cool = RUNS / "01_benchmark" / "tramp_cool" / "analysis" / "tramp.csv"
    have(bench, heat, cool)
    B = {r["phase"]: r for r in json.load(open(bench))}
    H = np.genfromtxt(heat, delimiter=",", names=True); Cc = np.genfromtxt(cool, delimiter=",", names=True)
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.9), gridspec_kw=dict(width_ratios=[0.8, 1.1, 1.1], wspace=0.38))
    # (a) 0 K phase energies
    ph = ["C", "T", "O", "R"]; dE = [B[p]["dE_meV_fu"] for p in ph]
    bars = ax[0].bar(ph, dE, color=[GREY, C[1], C[2], C[0]], width=0.62, edgecolor="white", linewidth=2)
    for b, p in zip(bars, ph):
        if p != "C":
            ax[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{B[p]['Pmag']:.2f}",
                       ha="center", va="bottom", fontsize=11, color="white")
    ax[0].set(ylabel=r"$\Delta E$ (meV/f.u.)", ylim=(min(dE) * 1.12, 3))
    ax[0].axhline(0, color="k", lw=1.0); label(ax[0], "a", y=0.97)
    NUM["phase_dE_meV_fu"] = {p: round(B[p]["dE_meV_fu"], 1) for p in ph}
    NUM["phase_P"] = {p: round(B[p]["Pmag"], 3) for p in ph}
    NUM["phase_lattice"] = {p: dict(a=round(B[p]["a"], 4), b=round(B[p]["b"], 4), c=round(B[p]["c"], 4),
                                    alpha=round(B[p]["alpha"], 2)) for p in ph}
    # (b) pseudo-cubic edges vs T
    for D, ls, mk, nm in ((H, "-", "o", "heating"), (Cc, "--", "s", "cooling")):
        for j, k in enumerate(("edge_lo", "edge_mid", "edge_hi")):
            ax[1].plot(D["T"], D[k], ls=ls, color=C[j], lw=1.8, label=f"{nm}" if j == 0 else None)
    ax[1].set(xlabel="Temperature (K)", ylabel=r"pseudo-cubic edge (Å)")
    # (c) mean local |P| per cell and net polarization
    for D, ls, nm in ((H, "-", "heating"), (Cc, "--", "cooling")):
        ax[2].plot(D["T"], D["absP"], ls=ls, color=C[0], lw=1.9)
        ax[2].plot(D["T"], D["netP"], ls=ls, color="k", lw=1.5)
    ax[2].set(xlabel="Temperature (K)", ylabel=r"polarization (C m$^{-2}$)", ylim=(0, None))
    leg = [Line2D([], [], color=C[0], lw=2, label=r"local $\langle|\mathbf{P}|\rangle$"),
           Line2D([], [], color="k", lw=1.6, label=r"net $|\langle\mathbf{P}\rangle|$"),
           Line2D([], [], color="0.55", lw=1.0, ls=":", label=r"expt. $T_c$"),
           Line2D([], [], color=GREY, ls="-", lw=1.6, label="heating"), Line2D([], [], color=GREY, ls="--", lw=1.6, label="cooling")]
    ax[2].set_ylim(0, ax[2].get_ylim()[1] * 1.35); ax[2].legend(handles=leg, ncol=3, loc="upper center", fontsize=9)
    ax[1].legend(handles=[Line2D([], [], color=C[j], lw=2, label=l) for j, l in enumerate((r"$a$ (shortest)", r"$b$", r"$c$ (longest)"))] +
                 leg[-3:], ncol=3, loc="upper center", fontsize=9)
    # phase of every frame from the space group of the averaged cell (analysis/phase_id.py), as bands under (b)
    PH = {160: ("R", C[0]), 38: ("O", C[2]), 99: ("T", C[1]), 221: ("C", "#b9b8b3")}
    SG_COL = "sg_005"   # symprec 0.05 A (genfromtxt drops the "."); 0.03 A flickers to Pm/Cm subgroups at finite T
    bands = {}
    for nm, run in (("heating", "tramp_heat"), ("cooling", "tramp_cool")):
        f = RUNS / "01_benchmark" / run / "analysis" / "phase_id.csv"
        if f.exists():
            Q = np.genfromtxt(f, delimiter=",", names=True, dtype=None, encoding=None)
            T_ = np.asarray(Q["T"], float); sg_ = np.asarray(Q[SG_COL] if SG_COL in Q.dtype.names else Q[SG_COL.replace("0", "0.", 1)], int)
            od = np.asarray(Q["order"], float)
            o = np.argsort(T_); bands[nm] = (T_[o], sg_[o], od[o])
    lo, hi = ax[1].get_ylim(); ax[1].set_ylim(lo, hi + 0.45 * (hi - lo))
    for x in ax[1:]:
        for t in EXP_T.values():
            x.axvline(t, color="0.55", lw=1.0, ls=":", zorder=0)
        x.set_xlim(0, 610)
    if bands:
        from matplotlib.transforms import blended_transform_factory
        tr = blended_transform_factory(ax[1].transData, ax[1].transAxes)
        lo, hi = ax[1].get_ylim(); ax[1].set_ylim(lo - 0.22 * (hi - lo), hi)
        win = {}
        for row, nm in enumerate(("heating", "cooling")):
            if nm not in bands:
                continue
            T_, sg_, od = bands[nm]; y0 = 0.10 - 0.075 * row
            mid = np.r_[T_[0], 0.5 * (T_[1:] + T_[:-1]), T_[-1]]
            labs = [PH.get(int(g), (None, None))[0] for g in sg_]
            multi = np.array([nm == "heating" and od[k] < 0.5 and T_[k] < 200 for k in range(len(T_))])
            # keep a phase only where it persists >= 3 consecutive frames; isolated frames count as intermediate
            keep = [None] * len(labs); k = 0
            while k < len(labs):
                m_ = k
                while m_ + 1 < len(labs) and labs[m_ + 1] == labs[k] and not multi[m_ + 1]:
                    m_ += 1
                if labs[k] and not multi[k] and m_ - k + 1 >= 3:
                    for q in range(k, m_ + 1):
                        keep[q] = labs[k]
                    if T_[m_] - T_[k] >= 25:            # letter at the centre of each window wide enough
                        ax[1].text(0.5 * (mid[k] + mid[m_ + 1]), y0 + 0.027, labs[k], transform=tr, fontsize=10,
                                   fontweight="bold", color="white", ha="center", va="center", zorder=3)
                k = m_ + 1
            colors = {l: c for l, c in PH.values()}
            for k in range(len(T_)):
                if multi[k]:
                    fc, hatch = "white", "////"
                else:
                    fc, hatch = (colors[keep[k]] if keep[k] else "white"), None
                ax[1].add_patch(plt.Rectangle((mid[k], y0), mid[k + 1] - mid[k], 0.055, transform=tr, facecolor=fc,
                                              edgecolor="0.6" if multi[k] else "none", lw=0, hatch=hatch, zorder=1))
                if keep[k]:
                    win.setdefault(nm, []).append((keep[k], float(T_[k])))
            ax[1].add_patch(plt.Rectangle((0, y0), 1, 0.055, transform=ax[1].transAxes, facecolor="none",
                                          edgecolor="0.6", lw=0.6, zorder=2))
            ax[1].text(1.012, y0 + 0.027, "heat" if nm == "heating" else "cool", transform=ax[1].transAxes,
                       fontsize=10, va="center", ha="left", clip_on=False)
        NUM["phase_id_windows"] = {nm: {ph: [min(t for p, t in w if p == ph), max(t for p, t in w if p == ph)]
                                        for ph in sorted({p for p, _ in w})} for nm, w in win.items()}
    label(ax[1], "b", x=0.9); label(ax[2], "c", x=0.03, y=0.15)
    save(fig, "Fig1")


# ============================================================================= Fig. 2: geometry + GB energies
def fig2():
    gbj = RUNS / "02_gb_energy" / "gb_energies.json"; have(gbj)
    rows = json.load(open(gbj))
    fig = plt.figure(figsize=(16.5, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.55, 1.35], wspace=0.35)
    # (a) Sigma5 interface registry: Ba sites of the last grain-1 and first grain-2 BaO layer
    axa = fig.add_subplot(gs[0])
    struct = ROOT / "structures" / "data" / "s5_210.data"
    try:
        from btoan.io import read_data
        D = read_data(struct); meta = json.load(open(str(struct) + ".json"))
        x = D["x"]; t = D["type"]; Ly = D["box"][1]; nc = meta["ncell_per_grain"]; a = meta["a"]
        ba = t == 1
        g1 = ba & (np.abs(x[:, 1] - (nc - 1) * a) < 0.3); g2 = ba & (np.abs(x[:, 1] - nc * a) < 0.3)
        axa.scatter(x[g1, 0], x[g1, 2], s=46, c=C[0], marker="o", label="grain 1", edgecolor="white", lw=0.8, zorder=3)
        axa.scatter(x[g2, 0], x[g2, 2], s=26, c=C[1], marker="s", label="grain 2", edgecolor="white", lw=0.6, zorder=4)
        L = D["box"][0]
        from scipy.spatial import cKDTree
        p1 = np.mod(x[g1][:, [0, 2]], L); p2 = np.mod(x[g2][:, [0, 2]], L)
        dd, _ = cKDTree(p2, boxsize=[L, L]).query(p1)
        cs = p1[dd < 0.3]
        axa.scatter(cs[:, 0], cs[:, 1], s=190, facecolors="none", edgecolors="k", lw=1.6, label="coincidence", zorder=5)
        NUM["sigma5_coincidence_fraction"] = round(float((dd < 0.3).mean()), 3)
        axa.set(xlim=(0, L), ylim=(0, L), aspect="equal", xlabel=r"$x$ (Å)", ylabel=r"$z$ (Å)")
        axa.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=10, frameon=False,
                   handletextpad=0.3, columnspacing=1.0)
    except Exception as e:                                     # structure file is optional for this panel
        axa.text(0.5, 0.5, f"structure not found\n{e}", ha="center", transform=axa.transAxes, fontsize=9)
    label(axa, "a", y=1.12)
    # (b) stacking along the normal (schematic)
    axb = fig.add_subplot(gs[1]); axb.set_xlim(0, 1); axb.set_ylim(0, 8); axb.axis("off")
    for i in range(8):                                   # continuous stacking: ...BaO TiO2 | BaO TiO2...
        lay = "BaO" if i % 2 == 0 else r"TiO$_2$"
        col = C[0] if i < 4 else C[1]
        axb.add_patch(plt.Rectangle((0.08, i + 0.12), 0.55, 0.76, color=col, alpha=0.18 if i % 2 else 0.38, lw=0))
        axb.text(0.355, i + 0.5, lay, ha="center", va="center", fontsize=11)
    axb.plot([0.02, 0.7], [4, 4], color="k", lw=1.6, ls="--"); axb.text(0.72, 4, "GB", va="center", fontsize=12)
    axb.annotate("", xy=(0.88, 0.8), xytext=(0.88, 3.2), arrowprops=dict(arrowstyle="-|>", lw=2.2, color="k"))
    axb.annotate("", xy=(0.88, 4.8), xytext=(0.88, 7.2), arrowprops=dict(arrowstyle="-|>", lw=2.2, color="k"))
    axb.text(0.88, 7.5, r"$P_y$", ha="center", fontsize=12)
    axb.text(0.355, 8.05, "grain 2", ha="center", fontsize=11, color=C[1]); axb.text(0.355, -0.45, "grain 1", ha="center", fontsize=11, color=C[0])
    axb.text(0.0, 2.0, r"$y$ (GB normal)", rotation=90, va="center", fontsize=12)
    label(axb, "b", y=1.12)
    # (c) gamma vs deviation from the nearest equivalent orientation
    axc = fig.add_subplot(gs[2])
    th = np.array([r["theta_total_deg"] for r in rows]); g_tot = np.array([r["gamma_J_m2"] for r in rows])
    # boundary energy with the grain-interior state difference removed (tools/gb_profile.py), if available
    pj = RUNS / "02_gb_energy" / "gb_profiles.json"
    loc = {q["boundary"]: q for q in json.load(open(pj))} if pj.exists() else {}
    for r in rows:
        q = loc.get(r["boundary"])
        r["gamma_loc"] = q["gamma_loc"] if q else None; r["D_int"] = q["D_int_meV_fu"] if q else None
    have_loc = all(r["gamma_loc"] is not None for r in rows)
    g = np.array([r["gamma_loc"] for r in rows]) if have_loc else g_tot
    dev = np.minimum(th, 90 - th); lo = th <= 45
    if have_loc:
        axc.scatter(dev, g_tot, s=40, facecolor="none", edgecolor="0.6", marker="D", lw=1.2, zorder=2,
                    label=r"$\gamma$ (total excess)")
    nm = r"$\gamma_{\rm loc}$, " if have_loc else ""
    axc.scatter(dev[lo], g[lo], s=70, c=C[0], marker="o", label=nm + r"$\theta\leq45^\circ$", zorder=3, edgecolor="white", lw=1)
    axc.scatter(dev[~lo], g[~lo], s=70, c=C[1], marker="s", label=nm + r"$\theta>45^\circ$ (at $90^\circ-\theta$)", zorder=3,
                edgecolor="white", lw=1)
    for r, d, gg in zip(rows, dev, g):
        h, k = r["hk"]
        if r["sigma"] == 61 or (r["sigma"] == 5 and (h, k) == (2, 1)):
            txt = r"$\Sigma5$(210), (310)" if r["sigma"] == 5 else r"$\Sigma61$(650)"
            axc.annotate(txt, (d, gg), xytext=(-60, 10) if r["sigma"] == 5 else (8, 4),
                         textcoords="offset points", fontsize=10)
    # equivalent pairs (theta, 90-theta): connect
    for s in sorted({r["sigma"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["sigma"] == s]
        if len(idx) == 2 and abs(dev[idx[0]] - dev[idx[1]]) < 0.1:
            axc.plot([dev[idx[0]]] * 2, [g[idx[0]], g[idx[1]]], color="0.6", lw=1.4, zorder=1)
    axc.set(xlabel=r"$\min(\theta,\,90^\circ-\theta)$ (deg)", ylabel=r"boundary energy (J m$^{-2}$)")
    axc.legend(loc="lower right", fontsize=9)
    label(axc, "c", y=1.12)
    pairs = [(r["sigma"], abs(g[i] - g[j])) for i, r in enumerate(rows) for j in range(i + 1, len(rows))
             if rows[j]["sigma"] == r["sigma"] and abs(dev[i] - dev[j]) < 0.1]
    NUM["gb_gamma_range"] = [round(float(g.min()), 3), round(float(g.max()), 3)]
    NUM["gb_pair_max_diff"] = round(max(p[1] for p in pairs), 3) if pairs else None
    NUM["gb_gamma"] = {r["boundary"]: r["gamma_J_m2"] for r in rows}
    if have_loc:
        NUM["gb_gamma_loc"] = {r["boundary"]: r["gamma_loc"] for r in rows}
        NUM["gb_gamma_total_range"] = [round(float(g_tot.min()), 3), round(float(g_tot.max()), 3)]
    save(fig, "Fig2")
    # table S: GB energies
    lines = [r"\begin{tabular}{lcccccc}", r"\hline", r"Boundary & $\theta$ (deg) & $\min(\theta,90^\circ-\theta)$ & "
             r"$\gamma$ (J\,m$^{-2}$) & $\Delta_{\rm int}$ (meV/f.u.) & $\gamma_{\rm loc}$ (J\,m$^{-2}$) & lowest-energy shift \\", r"\hline"]
    for r, d in sorted(zip(rows, dev), key=lambda q: q[0]["theta_total_deg"]):
        h, k = r["hk"]
        dl = "--" if r["D_int"] is None else f"{r['D_int']:+.1f}"; gl = "--" if r["gamma_loc"] is None else f"{r['gamma_loc']:.3f}"
        lines.append(fr"$\Sigma{r['sigma']}$({h}{k}0) & {r['theta_total_deg']:.2f} & {d:.2f} & {r['gamma_J_m2']:.3f} & {dl} & {gl} & "
                     + r["best_shift"].replace("shift_", "(").replace("_", ", ") + r") \\")
    lines += [r"\hline", r"\end{tabular}"]
    (OUT / "tab_gb_energies.tex").write_text("\n".join(lines) + "\n"); print("  wrote figures/out/tab_gb_energies.tex")


# ============================================================================= Fig. 3: single crystals
SC = [("[001]", "sc_001"), ("[011]", "sc_011"), ("[111]", "sc_111")]
R_PROJ = {"[001]": 1 / np.sqrt(3), "[011]": 2 / np.sqrt(6), "[111]": 1.0}   # max <111>.n


def fig3():
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.6), gridspec_kw=dict(hspace=0.32, wspace=0.3))
    P_R = NUM.get("phase_P", {}).get("R", 0.273)
    rows = {}
    for T, axP, axS in ((10, ax[0, 0], ax[0, 1]), (300, ax[0, 2], ax[1, 2])):
        for j, (lab, rid) in enumerate(SC):
            run = rd("03_single_crystal", T, f"{rid}_{T}K")
            if not (run / "analysis" / "frames.csv").exists():
                continue
            F = frames(run); m = loop_metrics(F["E"], F["P"], EMAX)
            rows[(lab, T)] = m
            axP.plot(F["E"], F["P"], color=C[j], ls=LS[j], lw=2.2, label=lab)
            if axS is not None:
                axS.plot(F["E"], F["s1"], color=C[j], ls=LS[j], lw=2.2, label=lab)
    for x, t in ((ax[0, 0], "10 K"), (ax[0, 2], "300 K")):
        x.set(xlabel=r"$E$ (V Å$^{-1}$)", ylabel=r"$P_\parallel$ (C m$^{-2}$)"); x.legend(loc="lower right", title=t)
    for x, t in ((ax[0, 1], "10 K"), (ax[1, 2], "300 K")):
        x.set(xlabel=r"$E$ (V Å$^{-1}$)", ylabel=r"$\varepsilon_\parallel$ (%)"); x.legend(loc="lower center", title=t, ncol=3, fontsize=9)
        x.set_ylim(x.get_ylim()[0] - 0.35 * np.ptp(x.get_ylim()), x.get_ylim()[1])
    for x in (ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 2]):
        axes0(x)
    # (d) Pr vs direction with the R-projection; (e) Ec
    labs = [l for l, _ in SC]; xi = np.arange(3); w = 0.36
    for k, (T, hatch, col) in enumerate(((10, None, C[0]), (300, "//", C[1]))):
        pr = [rows.get((l, T), {}).get("Pr", np.nan) for l in labs]
        ec = [rows.get((l, T), {}).get("Ec", np.nan) for l in labs]
        ax[1, 0].bar(xi + (k - 0.5) * w, pr, w, color=col, hatch=hatch, edgecolor="white", lw=1.5, label=f"{T} K")
        ax[1, 1].bar(xi + (k - 0.5) * w, ec, w, color=col, hatch=hatch, edgecolor="white", lw=1.5, label=f"{T} K")
    ax[1, 0].plot(xi - 0.5 * w, [P_R * R_PROJ[l] for l in labs], "kD", ms=9, mec="white", mew=1.2, label=r"R projection $|P_R|\cos\phi_{\min}$")
    for x, yl in ((ax[1, 0], r"$P_r$ (C m$^{-2}$)"), (ax[1, 1], r"$E_c$ (V Å$^{-1}$)")):
        x.set(xticks=xi, xticklabels=labs, ylabel=yl, xlabel="field direction")
        x.set_ylim(0, x.get_ylim()[1] * 1.4); x.legend(loc="upper left", ncol=1)
        x.tick_params(axis="x", which="minor", bottom=False, top=False)
    for x, s in zip((ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1], ax[1, 2]), "abcdef"):
        label(x, s)
    NUM["single_crystal"] = {f"{l}_{T}K": {k: round(float(v), 4) for k, v in m.items() if k in ("Ec", "Pr", "Ps", "squareness")}
                             for (l, T), m in rows.items()}
    save(fig, "Fig3")


# ============================================================================= Fig. 4: Sigma5 at 10 K
# lab-frame directions (y = GB normal); grain 1 has lab x = [201], y = [010], z = [-102]
S5_DIRS = [(r"$x$ (in plane)", "s5_100"), (r"$z$ (in plane)", "s5_001"), (r"$y$ (normal)", "s5_010"),
           (r"$y+z$", "s5_011"), (r"$x+z$ (in plane)", "s5_101"), (r"$x+y$", "s5_110"), (r"$x+y+z$", "s5_111")]


def fig4():
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 10), gridspec_kw=dict(hspace=0.3, wspace=0.28))
    m10 = {}
    groups = ((ax[0, 0], S5_DIRS[:3], "in-plane and normal"), (ax[0, 1], S5_DIRS[3:], "mixed"))
    for axp, lst, ttl in groups:
        ref = rd("03_single_crystal", 10, "sc_m102_10K")
        if (ref / "analysis" / "frames.csv").exists() and axp is ax[0, 0]:
            F = frames(ref); axp.plot(F["E"], F["P"], color="0.65", lw=3.2, label=r"crystal, $\hat n\parallel[\bar102]$", zorder=1)
        for j, (lab, rid) in enumerate(lst):
            run = rd("04_sigma5", 10, f"{rid}_10K")
            if not (run / "analysis" / "frames.csv").exists():
                continue
            F = frames(run); m10[lab] = loop_metrics(F["E"], F["P"], EMAX)
            axp.plot(F["E"], F["P"], color=C[j + (0 if axp is ax[0, 0] else 3)], ls=LS[j], lw=2.2, label=lab)
        axp.set(xlabel=r"$E$ (V Å$^{-1}$)", ylabel=r"interior $P_\parallel$ (C m$^{-2}$)"); axes0(axp)
        axp.legend(loc="lower right", title=r"$\Sigma5$, 10 K")
    # (c) layer profiles of Py and Pz at the two remanent states, field in plane (z)
    run = rd("04_sigma5", 10, "s5_001_10K")
    try:
        z, inner = cellp(run); P = z["P"]; k = pick(z["E"])
        for s, (key, ls) in enumerate((("rp", "-"), ("rm", "--"))):
            Pt = trem(P, k, key)
            yy, pz, Ly = layer_profile(z, Pt, 2); _, py, _ = layer_profile(z, Pt, 1)
            nm = r"$P_r^{+}$" if key == "rp" else r"$P_r^{-}$"
            mfc = None if key == "rp" else "white"
            ax[1, 0].plot(yy, pz, color=C[0], ls=ls, marker=MK[0], ms=6, mfc=mfc, lw=2, label=fr"$P_z$, {nm}")
            ax[1, 0].plot(yy, py, color=C[1], ls=ls, marker=MK[1], ms=6, mfc=mfc, lw=2, label=fr"$P_y$, {nm}")
        for gby in (0, Ly / 2, Ly):
            ax[1, 0].axvline(gby, color="k", ls=":", lw=1.4)
        ax[1, 0].set(xlabel=r"$y$ (Å)", ylabel=r"layer-averaged $P$ (C m$^{-2}$)", xlim=(0, Ly))
        ax[1, 0].axhline(0, color="0.75", lw=0.9)
        lo, hi = ax[1, 0].get_ylim(); ax[1, 0].set_ylim(lo, hi + 0.4 * (hi - lo))
        ax[1, 0].legend(ncol=4, loc="upper center", fontsize=9, columnspacing=0.8, handlelength=2.2)
        ax[1, 0].set_title(r"field along $z$ (in the GB plane)", fontsize=13)
    except FileNotFoundError as e:
        ax[1, 0].text(0.5, 0.5, str(e), ha="center", transform=ax[1, 0].transAxes, fontsize=8)
    # (d) field along the normal: Py profile at +Emax, -Emax and the two zero-field states
    run = rd("04_sigma5", 10, "s5_010_10K")
    try:
        z, inner = cellp(run); P = z["P"]; k = pick(z["E"])
        for j, (key, nm) in enumerate((("pmax", r"$+E_{\max}$"), ("rp", r"$E=0$ (desc.)"), ("mmax", r"$-E_{\max}$"),
                                       ("rm", r"$E=0$ (asc.)"))):
            yy, py, Ly = layer_profile(z, trem(P, k, key), 1)
            ax[1, 1].plot(yy, py, color=C[j], ls=LS[j], marker=MK[j], ms=6, lw=2, label=nm)
        for gby in (0, Ly / 2, Ly):
            ax[1, 1].axvline(gby, color="k", ls=":", lw=1.4)
        ax[1, 1].set(xlabel=r"$y$ (Å)", ylabel=r"layer-averaged $P_y$ (C m$^{-2}$)", xlim=(0, Ly))
        ax[1, 1].axhline(0, color="0.75", lw=0.9)
        lo, hi = ax[1, 1].get_ylim(); ax[1, 1].set_ylim(lo, max(hi, 0) + 0.35 * (hi - lo))
        ax[1, 1].legend(loc="upper center", ncol=4, fontsize=9, columnspacing=0.8, handlelength=2.2)
        ax[1, 1].set_title(r"field along $y$ (GB normal)", fontsize=13)
    except FileNotFoundError as e:
        ax[1, 1].text(0.5, 0.5, str(e), ha="center", transform=ax[1, 1].transAxes, fontsize=8)
    for x, s in zip(ax.flat, "abcd"):
        label(x, s, y=0.97 if s in "ab" else 1.1)
    NUM["sigma5_10K_interior"] = {l: {k: round(float(v), 4) for k, v in m.items()
                                      if k in ("Ec", "Pr", "Ps", "squareness", "P_offset", "Ec_imprint")} for l, m in m10.items()}
    save(fig, "Fig4")


# ============================================================================= Fig. 5: 300 K remanence loss + bias test
BIAS = [(0.0, "s5_001_300K"), (0.015, "s5b015_001_300K"), (0.030, "s5b030_001_300K"), (0.045, "s5b045_001_300K")]
# grain thickness (nm), label, run
THICK = [(4, r"4 nm", rd("04_sigma5", 300, "s5_001_300K")), (4, r"4 nm, 2$\times$ wide", rd("06_finite_size", 300, "s5W_001_300K")),
         (8, r"8 nm", rd("06_finite_size", 300, "s5T_001_300K")), (8, r"8 nm, 2$\times$ wide", rd("06_finite_size", 300, "s5L_001_300K")),
         (16, r"16 nm", rd("06_finite_size", 300, "s5T4_001_300K"))]


def fig5():
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 10), gridspec_kw=dict(hspace=0.36, wspace=0.32))
    ref = rd("03_single_crystal", 300, "sc_m102_300K")
    ref_pr = None
    out = {}
    # (a) 10 K vs 300 K, matched crystal at 300 K
    cases = [(r"crystal [$\bar1$02], 300 K", ref, GREY, "-", 3.0),
             (r"$\Sigma5$ (4 nm), 10 K", rd("04_sigma5", 10, "s5_001_10K"), C[0], "--", 2.2),
             (r"$\Sigma5$ (4 nm), 300 K", rd("04_sigma5", 300, "s5_001_300K"), C[1], "-.", 2.2),
             (r"$\Sigma61$, 300 K", rd("05_sigma61", 300, "s61_001_300K"), C[2], ":", 2.2)]
    for lab, run, col, ls, lw in cases:
        if not (run / "analysis" / "frames.csv").exists():
            continue
        F = frames(run); m = loop_metrics(F["E"], F["P"], EMAX); out[lab] = m
        if run == ref:
            ref_pr = m["Pr"]
        ax[0, 0].plot(F["E"], F["P"], color=col, ls=ls, lw=lw, label=lab)
    ax[0, 0].set(xlabel=r"$E_z$ (V Å$^{-1}$)", ylabel=r"interior $P_z$ (C m$^{-2}$)"); axes0(ax[0, 0]); ax[0, 0].legend(loc="lower right", fontsize=8, handlelength=1.8, borderpad=0.4)
    # (b) loops vs grain thickness, (c) Pr vs thickness
    pts = []
    for k, (t, lab, run) in enumerate(THICK):
        if not (run / "analysis" / "frames.csv").exists():
            continue
        F = frames(run); m = loop_metrics(F["E"], F["P"], EMAX)
        z, inner = cellp(run); kk = pick(z["E"]); py = np.mean([trem(z["P"], kk, q)[inner, 1].mean() for q in ("rp", "rm")])
        pts.append((t, m["Pr"], py, "wide" in lab)); out["thick " + lab] = m
        ax[0, 1].plot(F["E"], F["P"], color=C[k], ls=LS[k], lw=2.0, label=lab)
    ax[0, 1].set(xlabel=r"$E_z$ (V Å$^{-1}$)", ylabel=r"interior $P_z$ (C m$^{-2}$)"); axes0(ax[0, 1])
    ax[0, 1].legend(loc="upper left", fontsize=8, title=r"$\Sigma5$, 300 K", title_fontsize=9, handlelength=1.8, borderpad=0.4)
    if pts:
        P_ = np.array([(t, pr, py) for t, pr, py, w in pts]); wide = np.array([w for *_, w in pts])
        ax[0, 2].plot(P_[~wide, 0], P_[~wide, 1], "o-", color=C[0], ms=9, lw=2.2, label=r"$\Sigma5$, standard width")
        if wide.any():
            ax[0, 2].plot(P_[wide, 0], P_[wide, 1], "s", color=C[1], ms=9, mfc="white", mew=2, label=r"$\Sigma5$, 2$\times$ width")
        if ref_pr is not None:
            ax[0, 2].axhline(ref_pr, color=GREY, ls="--", lw=1.8, label="matched crystal")
        ax[0, 2].set(xlabel="grain thickness (nm)", ylabel=r"$P_r$ (C m$^{-2}$)", xscale="log", ylim=(0, None),
                     xticks=[4, 8, 16], xticklabels=["4", "8", "16"]); ax[0, 2].legend(loc="lower right", fontsize=9)
        ax[0, 2].xaxis.set_minor_formatter(plt.NullFormatter()); ax[0, 2].tick_params(axis="x", which="minor", bottom=False, top=False)
        NUM["thickness_300K"] = [dict(nm=int(t), Pr=round(float(pr), 4), Py_E0=round(float(py), 4), wide=bool(w)) for t, pr, py, w in pts]
    # (d) bias loops, (e) Pr vs bias, (f) <Py> vs bias
    rows = []
    for j, (eb, rid) in enumerate(BIAS):
        run = rd("04_sigma5", 300, rid)
        if not (run / "analysis" / "frames.csv").exists():
            continue
        F = frames(run); m = loop_metrics(F["E"], F["P"], EMAX)
        z, inner = cellp(run); kk = pick(z["E"])
        py = np.mean([trem(z["P"], kk, q)[inner, 1].mean() for q in ("rp", "rm")])
        rows.append((eb, m["Pr"], m["Ec"], py))
        ax[1, 0].plot(F["E"], F["P"], color=C[j], ls=LS[j], lw=2.2, label=fr"{eb:.3f}")
    ax[1, 0].set(xlabel=r"$E_z$ (V Å$^{-1}$)", ylabel=r"interior $P_z$ (C m$^{-2}$)"); axes0(ax[1, 0])
    ax[1, 0].legend(loc="lower right", fontsize=8, title=r"$E_y^{\rm bias}$ (V Å$^{-1}$)", title_fontsize=9, handlelength=1.8, borderpad=0.4)
    if rows:
        R = np.array(rows)
        ax[1, 1].plot(R[:, 0], R[:, 1], "o-", color=C[0], ms=9, lw=2.2, label=r"$\Sigma5$ (4 nm)")
        if ref_pr is not None:
            ax[1, 1].axhline(ref_pr, color=GREY, ls="--", lw=1.8, label="matched crystal")
        ax[1, 1].set(xlabel=r"static bias $E_y^{\rm bias}$ (V Å$^{-1}$)", ylabel=r"$P_r$ (C m$^{-2}$)", ylim=(0, None))
        ax[1, 1].legend(loc="center right", fontsize=9)
        ax[1, 2].plot(R[:, 0], R[:, 3], "s-", color=C[1], ms=9, lw=2.2); ax[1, 2].axhline(0, color="0.75", lw=0.9)
        ax[1, 2].set(xlabel=r"static bias $E_y^{\rm bias}$ (V Å$^{-1}$)", ylabel=r"$\langle P_y\rangle_{E=0}$ (C m$^{-2}$)")
        NUM["bias_test"] = [dict(E_bias=float(e), Pr=round(float(p), 4), Ec=round(float(c), 4), Py_E0=round(float(y), 4)) for e, p, c, y in R]
    for x, s_ in zip(ax.flat, "abcdef"):
        label(x, s_)
    NUM["T300_inplane"] = {l: {k: round(float(v), 4) for k, v in m.items() if k in ("Ec", "Pr", "Ps", "squareness")} for l, m in out.items()}
    save(fig, "Fig5")


def figS_inst():
    runs = [(r"$\Sigma5$ (4 nm), 10 K", rd("04_sigma5", 10, "s5_001_10K")), (r"$\Sigma5$ (4 nm), 300 K", rd("04_sigma5", 300, "s5_001_300K")),
            (r"$\Sigma5$ (16 nm), 300 K", rd("06_finite_size", 300, "s5T4_001_300K"))]
    runs = [(l, r) for l, r in runs if (r / "analysis" / "cellP.npz").exists()]
    have(*[r / "analysis" / "cellP.npz" for _, r in runs[:1]])
    fig, ax = plt.subplots(2, len(runs), figsize=(5.5 * len(runs), 9), squeeze=False, gridspec_kw=dict(hspace=0.35, wspace=0.3))
    for c, (lab, run) in enumerate(runs):
        z, m = cellp(run); P, E = z["P"], z["E"]
        rng = np.random.default_rng(0); cells = rng.choice(np.where(m)[0], 5, replace=False)
        t = np.arange(len(E)) * 0.8
        for k, cc in enumerate(cells):
            ax[0, c].plot(t, P[:, cc, 2], lw=1.0, color=C[k])
        ax[0, c].plot(t, E / np.nanmax(np.abs(E)) * 0.3, color="k", ls="--", lw=1.6, label=r"$0.3\,E/E_{\max}$")
        ax[0, c].set(xlabel="time in cycle (ps)", ylabel=r"per-cell $P_z$ (C m$^{-2}$)", title=lab); ax[0, c].legend(loc="lower left", fontsize=9)
        ax[1, c].plot(E, np.abs(P[:, m, 2]).mean(1), color=C[0], lw=2, label=r"$\langle|P_z|\rangle$")
        ax[1, c].plot(E, np.abs(P[:, m, 2].mean(1)), color=C[1], ls="--", lw=2, label=r"$|\langle P_z\rangle|$")
        ax[1, c].plot(E, np.abs(P[:, m, 1]).mean(1), color=C[2], ls=":", lw=2.4, label=r"$\langle|P_y|\rangle$")
        ax[1, c].set(xlabel=r"$E_z$ (V Å$^{-1}$)", ylabel=r"instantaneous average (C m$^{-2}$)", ylim=(0, None)); ax[1, c].legend(fontsize=9)
        label(ax[0, c], "abc"[c]); label(ax[1, c], "def"[c])
    save(fig, "FigS_inst")


def figS_term():
    """termination and electrostatics controls: Wolf vs PPPM, continuous vs reversed stacking."""
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(wspace=0.3))
    out = {}
    sets = [(ax[0], [("TiO$_2$|BaO, Wolf", "s5_010_10K"), ("TiO$_2$|BaO, PPPM", "s5p_010_10K"), ("BaO|TiO$_2$, Wolf", "s5r_010_10K")],
             r"$E_y$ (V Å$^{-1}$)", r"interior $P_y$ (C m$^{-2}$)", r"$\Sigma5$, 10 K, field along $y$"),
            (ax[1], [("TiO$_2$|BaO, Wolf", "s5_001_10K"), ("BaO|TiO$_2$, Wolf", "s5r_001_10K")],
             r"$E_z$ (V Å$^{-1}$)", r"interior $P_z$ (C m$^{-2}$)", r"$\Sigma5$, 10 K, field along $z$")]
    for axp, lst, xl, yl, ttl in sets:
        for j, (lab, rid) in enumerate(lst):
            run = rd("04_sigma5", 10, rid)
            if not (run / "analysis" / "frames.csv").exists():
                continue
            F = frames(run); m = loop_metrics(F["E"], F["P"], EMAX); out[rid] = m
            axp.plot(F["E"], F["P"], color=C[j], ls=LS[j], lw=2.2, label=lab)
        axp.set(xlabel=xl, ylabel=yl); axes0(axp); axp.legend(loc="upper left", fontsize=9, title=ttl)
    f = RUNS / "08_termination" / "term_test.json"
    if f.exists():
        R = json.load(open(f))
        names = {"cont_wolf": ("TiO$_2$|BaO, Wolf", 0, "-"), "cont_wolf_seedP": ("TiO$_2$|BaO, Wolf, $+P_y$ seed", 0, ":"),
                 "rev_wolf": ("BaO|TiO$_2$, Wolf", 2, "-"), "cont_pppm": ("TiO$_2$|BaO, PPPM", 1, "--"),
                 "cont_pppm_seedP": ("TiO$_2$|BaO, PPPM, $+P_y$ seed", 1, (0, (1, 1))), "rev_pppm": ("BaO|TiO$_2$, PPPM", 3, "--")}
        for k, (lab, ci, ls) in names.items():
            if k not in R:
                continue
            pr = np.array(R[k]["Py_profile"]); nb = len(pr); ok = pr != 0.0       # empty bins are stored as exactly 0
            ax[2].plot(((np.arange(nb) + 0.5) / nb)[ok], pr[ok], color=C[ci], ls=ls, marker=MK[ci], ms=4, lw=1.8, label=lab)
        for g in (0, 0.5, 1):
            ax[2].axvline(g, color="k", ls=":", lw=1.4)
        ax[2].axhline(0, color="0.75", lw=0.9)
        ax[2].set(xlabel=r"$y/L_y$", ylabel=r"layer-averaged $P_y$ (C m$^{-2}$)", xlim=(0, 1))
        lo, hi = ax[2].get_ylim(); ax[2].set_ylim(lo, hi + 0.45 * (hi - lo)); ax[2].legend(ncol=2, fontsize=8, loc="upper center")
        ax[2].set_title("0 K relaxation", fontsize=13)
        NUM["term_test_0K"] = {k: dict(P_g1=v["P_g1"], P_g2=v["P_g2"], Py_interior=v.get("Py_interior")) for k, v in R.items()}
    else:
        ax[2].text(0.5, 0.5, "runs/08_termination/term_test.json\nnot found", ha="center", transform=ax[2].transAxes, fontsize=9)
    for x, s_ in zip(ax, "abc"):
        label(x, s_)
    NUM["termination_loops"] = {k: {q: round(float(v), 4) for q, v in m.items() if q in ("Ec", "Pr", "Ps", "P_offset")} for k, m in out.items()}
    save(fig, "FigS_term")


# ============================================================================= Fig. 6: strain
def fig6():
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5), gridspec_kw=dict(wspace=0.28))
    for x, T in zip(ax, (10, 300)):
        ref = rd("03_single_crystal", T, f"sc_m102_{T}K"); run = rd("04_sigma5", T, f"s5_001_{T}K")
        if (ref / "analysis" / "frames.csv").exists():
            F = frames(ref); x.plot(F["E"], F["s1"], color=GREY, lw=3, label=r"crystal, $\hat n\parallel[\bar102]$")
        if (run / "analysis" / "frames.csv").exists():
            F = frames(run)
            x.plot(F["E"], F["s1"], color=C[0], ls="-", lw=2.2, label="grain 1")
            x.plot(F["E"], F["s2"], color=C[1], ls="--", lw=2.2, label="grain 2")
        x.set(xlabel=r"$E_z$ (V Å$^{-1}$)", ylabel=r"$\varepsilon_{zz}$ (%)"); axes0(x)
        lo, hi = x.get_ylim(); x.set_ylim(lo, hi + 0.3 * (hi - lo))
        x.legend(loc="upper center", title=fr"field along $z$, {T} K", ncol=3, fontsize=9)
    label(ax[0], "a"); label(ax[1], "b")
    save(fig, "Fig6")


# ============================================================================= SI figures
def figS_size():
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 10), gridspec_kw=dict(wspace=0.28, hspace=0.34))
    for j, (lab, run) in enumerate(((r"$\Sigma5$ (35.9$\times$80.2$\times$35.9 Å$^3$)", rd("04_sigma5", 10, "s5_001_10K")),
                                    (r"$\Sigma5$-L (71.7$\times$160.4$\times$71.7 Å$^3$)", rd("06_finite_size", 10, "s5L_001_10K")))):
        if not (run / "analysis" / "frames.csv").exists():
            continue
        F = frames(run); ax[0, 0].plot(F["E"], F["P"], color=C[j], ls=LS[j], lw=2.2, label=lab)
        z, inner = cellp(run); k = pick(z["E"]); yy, py, Ly = layer_profile(z, trem(z["P"], k, "rp"), 1)
        ax[0, 1].plot(yy / Ly, py, color=C[j], ls=LS[j], marker=MK[j], ms=5, lw=2, label=lab)
    ax[0, 0].set(xlabel=r"$E_z$ (V Å$^{-1}$)", ylabel=r"interior $P_z$ (C m$^{-2}$)", title="10 K"); axes0(ax[0, 0])
    ax[0, 0].legend(fontsize=9, loc="lower right")
    ax[0, 1].set(xlabel=r"$y/L_y$", ylabel=r"layer-averaged $P_y$ at $P_r^{+}$ (C m$^{-2}$)", title="10 K"); ax[0, 1].axhline(0, color="0.75", lw=0.9)
    ax[0, 1].legend(fontsize=9, loc="center")
    # 300 K grain-thickness series: layer profiles of P_z and P_y at P_r+
    for j, (t, lab, run) in enumerate([x for x in THICK if "wide" not in x[1]]):
        if not (run / "analysis" / "cellP.npz").exists():
            continue
        z, inner = cellp(run); k = pick(z["E"]); Pt = trem(z["P"], k, "rp")
        yy, pz, Ly = layer_profile(z, Pt, 2); _, py, _ = layer_profile(z, Pt, 1)
        ax[1, 0].plot(yy / Ly, pz, color=C[j], ls=LS[j], marker=MK[j], ms=4, lw=1.8, label=lab)
        ax[1, 1].plot(yy / Ly, py, color=C[j], ls=LS[j], marker=MK[j], ms=4, lw=1.8, label=lab)
    for x, comp in ((ax[1, 0], "z"), (ax[1, 1], "y")):
        x.set(xlabel=r"$y/L_y$", ylabel=fr"layer-averaged $P_{comp}$ at $P_r^{{+}}$ (C m$^{{-2}}$)", title=r"300 K, field along $z$")
        x.axhline(0, color="0.75", lw=0.9); x.legend(fontsize=9, title="grain thickness", loc="best")
    for x in ax.flat:
        for g in (0, 0.5, 1):
            x.axvline(g, color="k", ls=":", lw=1.4) if x is not ax[0, 0] else None
    for x, s_ in zip(ax.flat, "abcd"):
        label(x, s_)
    save(fig, "FigS_size")


def figS_dw():
    f = RUNS / "07_domain_walls" / "dw_energies.json"; have(f)
    rows = [r for r in json.load(open(f)) if "profile_meV_fu" in r]
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6), sharey=True, gridspec_kw=dict(wspace=0.08))
    for x, w in zip(ax, ("71", "109", "180")):
        for j, suf in enumerate(("", "_2x", "_long", "_2x_long")):
            r = next((q for q in rows if q["wall"] == w + suf), None)
            if r is None:
                continue
            p = np.array(r["profile_meV_fu"]); s = (np.arange(len(p)) + 0.5) / len(p)
            x.plot(s, p, color=C[j], ls=LS[j], marker=MK[j], ms=4, lw=1.8,
                   label={"": "40 Å", "_2x": "80 Å", "_long": "40 Å, 4× relax", "_2x_long": "80 Å, 4× relax"}[suf])
        x.axhline(0, color="0.75", lw=0.9)
        for g in (0, 0.5, 1):
            x.axvline(g, color="k", ls=":", lw=1.2)
        x.set(xlabel=r"$y/L_y$", title=f"{w}° wall")
    ax[0].set_ylabel(r"$E - E_{\rm mono}$ (meV/f.u.)"); ax[0].legend(fontsize=9)
    save(fig, "FigS_dw")
    lines = [r"\begin{tabular}{lcccc}", r"\hline", r"Wall & spacing & relaxation & $\sigma_{\rm DW}$ (mJ\,m$^{-2}$) & "
             r"angle between domains \\", r"\hline"]
    for r in json.load(open(f)):
        sp = r"80~\AA" if "_2x" in r["wall"] else r"40~\AA"; rl = "4$\\times$" if "_long" in r["wall"] else "1$\\times$"
        lines.append(fr"{r['wall'].split('_')[0]}$^\circ$ & {sp} & {rl} & {r['sigma_mJ_m2']:.1f} & {r['angle_between_domains']:.1f}$^\circ$ \\")
    lines += [r"\hline", r"\end{tabular}"]
    (OUT / "tab_dw.tex").write_text("\n".join(lines) + "\n"); print("  wrote figures/out/tab_dw.tex")
    NUM["domain_walls"] = {r["wall"]: r["sigma_mJ_m2"] for r in json.load(open(f))}


def table_loops():
    """loop metrics (grain interior) for every run with frames.csv."""
    lines = [r"\begin{tabular}{llcccccc}", r"\hline", r"System & $\hat n$ & $T$ (K) & $E_c$ (V\,\AA$^{-1}$) & $P_r$ & $P_s$ & "
             r"$P_r/P_s$ & offset \\", r"\hline"]
    found = 0
    for f in sorted(RUNS.glob("0[3-6]_*/*K/*/analysis/frames.csv")):
        run = f.parents[1]; prm = json.load(open(run / "params.json")) if (run / "params.json").exists() else {}
        try:
            m = metrics_interior(run)
        except Exception:
            continue
        n = "[" + "".join(str(int(float(prm.get(k, 0)))) for k in ("dx", "dy", "dz")) + "]"
        nm = run.name.replace("_", r"\_")
        ec = "--" if not np.isfinite(m["Ec"]) else f"{m['Ec']:.4f}"
        lines.append(fr"\texttt{{{nm}}} & {n} & {prm.get('T', '?')} & {ec} & {m['Pr']:.3f} & "
                     fr"{m['Ps']:.3f} & {m['squareness']:.2f} & {m['P_offset']:+.3f} \\")
        found += 1
    lines += [r"\hline", r"\end{tabular}"]
    (OUT / "tab_loops.tex").write_text("\n".join(lines) + "\n"); print(f"  wrote figures/out/tab_loops.tex ({found} runs)")


def figS_hist():
    """instantaneous per-cell polarization in the single frame closest to E = 0 after +Emax; field-aligned frame for all rows."""
    runs = [(r"$\Sigma5$ (4 nm), 10 K", rd("04_sigma5", 10, "s5_001_10K")), (r"$\Sigma5$ (4 nm), 300 K", rd("04_sigma5", 300, "s5_001_300K")),
            (r"$\Sigma5$ (16 nm), 300 K", rd("06_finite_size", 300, "s5T4_001_300K")),
            (r"crystal [$\bar1$02], 300 K", rd("03_single_crystal", 300, "sc_m102_300K"))]
    runs = [(l, r) for l, r in runs if (r / "analysis" / "cellP.npz").exists()]
    have(*[r / "analysis" / "cellP.npz" for _, r in runs[:1]])
    fig, ax = plt.subplots(len(runs), 3, figsize=(16, 4.3 * len(runs)), squeeze=False, gridspec_kw=dict(hspace=0.45, wspace=0.3))
    bins = np.linspace(-0.45, 0.45, 61); out = {}
    for r_, (lab, run) in enumerate(runs):
        z, m = cellp(run)
        if "crystal" in lab:
            m = np.ones(z["P"].shape[1], bool)
        k = pick(z["E"]); p = z["P"][k["rp"]][m]                  # ONE frame: the frame closest to E = 0 after +Emax
        e0 = float(z["E"][k["rp"]])
        if "crystal" in lab:
            # crystal axes -> the field-aligned frame of Sigma5 grain 1: x = [201]/sqrt5, y = [010], z = [-102]/sqrt5 (= field)
            Rm = np.array([[2, 0, 1], [0, 5 ** 0.5, 0], [-1, 0, 2]]) / 5 ** 0.5
            p = p @ Rm.T
        for c, (nm, ls) in enumerate(((r"$P_x$", ":"), (r"$P_y$", "--"), (r"$P_z$", "-"))):
            ax[r_, 0].hist(p[:, c], bins=bins, histtype="step", lw=2, color=C[c], ls=ls, label=nm, density=True)
        ax[r_, 0].set(xlabel=r"per-cell component (C m$^{-2}$)", ylabel="probability density", title=lab); ax[r_, 0].legend(fontsize=9)
        a = np.linalg.norm(p, axis=1)
        ax[r_, 1].hist(a, bins=np.linspace(0, 0.5, 51), histtype="stepfilled", color=C[0], alpha=0.6, density=True)
        ax[r_, 1].axvline(0.273, color="k", ls=":", lw=1.4, label=r"$|\mathbf{P}_R|$ (0 K)")
        ax[r_, 1].set(xlabel=r"per-cell $|\mathbf{P}|$ (C m$^{-2}$)", ylabel="probability density"); ax[r_, 1].legend(fontsize=9)
        ax[r_, 2].hist2d(p[:, 2], p[:, 1], bins=[bins, bins], cmap="Blues")
        ax[r_, 2].axhline(0, color="0.6", lw=0.8); ax[r_, 2].axvline(0, color="0.6", lw=0.8)
        ax[r_, 2].set(xlabel=r"per-cell $P_z$ (C m$^{-2}$)", ylabel=r"per-cell $P_y$ (C m$^{-2}$)", aspect="equal")
        for c_, s_ in enumerate("abc"):
            label(ax[r_, c_], f"{s_}{r_ + 1}")
        pz = p[:, 2]; mu = pz.mean(); sd = pz.std(); sk = ((pz - mu) ** 3).mean() / sd ** 3; ku = ((pz - mu) ** 4).mean() / sd ** 4
        out[lab] = dict(E_frame=round(e0, 5), n_cells=int(len(p)), mean=np.round(p.mean(0), 4).tolist(), mean_abs=np.round(np.abs(p).mean(0), 4).tolist(), absP=round(float(a.mean()), 4),
                        std=np.round(p.std(0), 4).tolist(), frac_Pz_gt_0p1=round(float((pz > 0.1).mean()), 3),
                        frac_Pz_lt_m0p1=round(float((pz < -0.1).mean()), 3), bimodality_coeff_Pz=round(float((sk ** 2 + 1) / ku), 3))
        print(f"  {run.name}: E={e0:+.4f} n={len(p)} <P>={out[lab]['mean']} <|P_i|>={out[lab]['mean_abs']} <|P|>={out[lab]['absP']} "
              f"f(Pz>0.1)={out[lab]['frac_Pz_gt_0p1']} f(Pz<-0.1)={out[lab]['frac_Pz_lt_m0p1']} BC(Pz)={out[lab]['bimodality_coeff_Pz']}")
    NUM["hist_Pr_plus_instantaneous"] = out
    save(fig, "FigS_hist")


def table_checks():
    """SI tables from analysis/referee_checks.py output: interior-margin test and core-shell separations."""
    f = OUT / "referee_margins.json"
    if f.exists():
        R = json.load(open(f))
        L = [r"\begin{tabular}{lccccccc}", r"\hline", r"Run & exclusion & cells & $E_c$ (V\,\AA$^{-1}$) & $P_r$ & $P_s$ & offset & $\langle P_y\rangle_{P_r^+}$ \\", r"\hline"]
        for run, d in R.items():
            for w, q in d.items():
                ww = r"$0.05\,L_y$" if "Ly" in w else fr"{w}~\AA"
                L.append(fr"\texttt{{{run.replace('_', chr(92) + '_')}}} & {ww} & {q['n_cells']} & {q['Ec']:.4f} & {q['Pr']:.3f} & {q['Ps']:.3f} & {q['offset']:+.3f} & {q['Py_Pr']:+.3f} \\")
            L.append(r"\hline")
        L += [r"\end{tabular}"]
        (OUT / "tab_margins.tex").write_text("\n".join(L) + "\n"); print("  wrote figures/out/tab_margins.tex")
    f = OUT / "referee_csep.json"
    if f.exists():
        R = json.load(open(f))
        L = [r"\begin{tabular}{lccc}", r"\hline", r"Run & status & max.\ separation (\AA) & max.\ 99.9th percentile (\AA) \\", r"\hline"]
        for run, q in R.items():
            mx = "--" if q["max_sep_A"] is None else f"{q['max_sep_A']:.3f}"; pc = "--" if q["p999_max_A"] is None else f"{q['p999_max_A']:.3f}"
            L.append(fr"\texttt{{{run.replace('_', chr(92) + '_')}}} & {q['status'].replace('_', ' ')} & {mx} & {pc} \\")
        L += [r"\hline", r"\end{tabular}"]
        (OUT / "tab_csep.tex").write_text("\n".join(L) + "\n"); print("  wrote figures/out/tab_csep.tex")


FIGS = {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5, "6": fig6, "S_size": figS_size, "S_dw": figS_dw, "S_inst": figS_inst, "S_term": figS_term, "S_hist": figS_hist, "checks": table_checks,
        "tables": table_loops}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*", default=list(FIGS)); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "numbers.json").exists():
        NUM.update(json.load(open(OUT / "numbers.json")))
    for key in a.only:
        print(f"Fig {key}:")
        try:
            FIGS[key]()
        except FileNotFoundError as e:
            print(f"  skipped -- {e}")
        except Exception:
            print("  FAILED:"); traceback.print_exc()
    json.dump(NUM, open(OUT / "numbers.json", "w"), indent=1)
    print("wrote figures/out/numbers.json")
