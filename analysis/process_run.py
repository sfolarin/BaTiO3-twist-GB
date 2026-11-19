#!/usr/bin/env python3
"""
Process one run folder -> analysis/ subfolder with:
  metrics.json   loop metrics (global and per grain), run parameters
  loop.csv       step, E, P_par (global, monitor), strain_par (global box)
  frames.csv     per dump frame of the cycle: E, mean cell P along n (all / grain1 / grain2 / GB cells),
                 grain strain along n (grain1 / grain2), switched fraction, mean |P_cell|, mean cos(P_cell,n)
  cellP.npz      per-cell polarization for every cycle frame (+ cell positions, grain labels)
  quicklook.png  4-panel overview

  python analysis/process_run.py runs/04_sigma5/10K/s5_001_10K
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from btoan.io import read_traj, read_data, iter_dump, load_params
from btoan.polarization import build_cells, cell_polarization
from btoan.strain import grain_masks, fit_F, small_strain, strain_along, box_strain_along
from btoan.loops import loop_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--n", type=float, nargs=3, help="field direction (default: params.json)")
    ap.add_argument("--Emax", type=float, default=0.06)
    ap.add_argument("--margin", type=float, default=0.05, help="GB exclusion margin (fraction of Ly)")
    ap.add_argument("--data", default="input.data")
    a = ap.parse_args()

    rd = Path(a.run_dir); out = rd / "analysis"; out.mkdir(exist_ok=True)
    prm = load_params(rd)
    n = np.array(a.n if a.n else [float(prm["dx"]), float(prm["dy"]), float(prm["dz"])], float)
    n /= np.linalg.norm(n)

    # ---- global loop from traj.dat ------------------------------------------------
    tr = read_traj(rd / "traj.dat")
    cyc = tr["stage"] == "cycle"
    E, P = tr["E"][cyc], tr["Ppar"][cyc]
    L = np.stack([tr["lx"], tr["ly"], tr["lz"]], 1)[cyc]
    eps_g = box_strain_along(L, L[0], n)
    np.savetxt(out / "loop.csv", np.column_stack([tr["step"][cyc], E, P, eps_g]), delimiter=",",
               header="step,E_V_per_A,Ppar_C_m2,strain_par_box", comments="")
    m_global = loop_metrics(E, P, a.Emax)
    step2E = dict(zip(tr["step"].astype(int), tr["E"]))
    step2stage = dict(zip(tr["step"].astype(int), tr["stage"]))

    # ---- topology + grains from the as-built structure -------------------------------
    D = read_data(rd / a.data)
    cells = build_cells(D["type"], D["q"], D["x"], D["box"], D.get("csid"))
    is_bi = D["meta"].get("kind", "").startswith("twist")
    yfrac = cells["ti_ref"][:, 1] / D["box"][1]
    if is_bi:
        g1, g2 = grain_masks(yfrac, margin_frac=a.margin)
        gbcell = ~(g1 | g2)
    else:
        g1 = np.ones(len(yfrac), bool); g2 = np.zeros(len(yfrac), bool); gbcell = np.zeros(len(yfrac), bool)
    ti_idx = cells["ti"]
    # lattice markers for strain: Ba cores (A-site; no polar off-centering, unlike Ti)
    ba_idx = np.where(D["type"] == 1)[0]
    yba = np.mod(D["x"][ba_idx, 1], D["box"][1]) / D["box"][1]
    if is_bi:
        b1, b2 = grain_masks(yba, margin_frac=a.margin)
    else:
        b1 = np.ones(len(ba_idx), bool); b2 = np.zeros(len(ba_idx), bool)

    # ---- per-frame analysis over the cycle -------------------------------------------
    rows, Pcells, Xref = [], [], None
    for fr in iter_dump(rd / "dump.lammpstrj"):
        if step2stage.get(fr["step"]) != "cycle":
            continue
        x = fr["x"]
        if Xref is None:
            Xref = x[ba_idx].copy()             # strain reference: first cycle frame (poled, E = 0)
        pc = cell_polarization(cells, x, fr["H"])
        ppar = pc @ n
        row = [fr["step"], step2E.get(fr["step"], np.nan), ppar.mean(),
               ppar[g1].mean(), ppar[g2].mean() if g2.any() else np.nan,
               ppar[gbcell].mean() if gbcell.any() else np.nan]
        for g in (b1, b2):
            if g.any():
                eps = small_strain(fit_F(Xref[g], x[ba_idx][g]))
                row.append(strain_along(eps, n))
            else:
                row.append(np.nan)
        mag = np.linalg.norm(pc, axis=1)
        row += [(ppar > 0).mean(), mag.mean(), (ppar / np.maximum(mag, 1e-12)).mean()]
        rows.append(row); Pcells.append(pc.astype(np.float32))
    cols = ["step", "E", "Pcell_par_all", "Pcell_par_g1", "Pcell_par_g2", "Pcell_par_gb",
            "strain_par_g1", "strain_par_g2", "frac_switched_pos", "mean_abs_Pcell", "mean_cos_to_n"]
    F = np.array(rows)
    np.savetxt(out / "frames.csv", F, delimiter=",", header=",".join(cols), comments="")
    np.savez_compressed(out / "cellP.npz", P=np.array(Pcells), steps=F[:, 0], E=F[:, 1],
                        ti_ref=cells["ti_ref"], grain1=g1, grain2=g2, gb=gbcell, n=n)

    metrics = dict(run=prm.get("run_id", rd.name), n=n.tolist(), T=prm.get("T"), global_loop=m_global,
                   n_frames=len(F), n_cells=int(len(ti_idx)), n_gb_cells=int(gbcell.sum()))
    if len(F) > 10:
        for key, col in (("cell_all", 2), ("grain1", 3), ("grain2", 4), ("gb_cells", 5)):
            if np.isfinite(F[:, col]).all():
                metrics[f"loop_{key}"] = loop_metrics(F[:, 1], F[:, col], a.Emax)
    json.dump(metrics, open(out / "metrics.json", "w"), indent=1)

    # ---- quicklook --------------------------------------------------------------------
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0, 0].plot(E, P, lw=1); ax[0, 0].set(xlabel="E (V/Å)", ylabel="P∥ monitor (C/m²)", title="global loop (traj.dat)")
    ax[0, 1].plot(F[:, 1], F[:, 3], label="grain 1"); ax[0, 1].plot(F[:, 1], F[:, 4], "--", label="grain 2")
    ax[0, 1].plot(F[:, 1], F[:, 5], ":", label="GB cells"); ax[0, 1].legend(); ax[0, 1].set(xlabel="E (V/Å)", ylabel="cell P∥ (C/m²)", title="per-cell polarization")
    ax[1, 0].plot(E, eps_g * 100); ax[1, 0].set(xlabel="E (V/Å)", ylabel="box strain ∥ (%)", title="global strain along n")
    ax[1, 1].plot(F[:, 1], F[:, 6] * 100, label="grain 1"); ax[1, 1].plot(F[:, 1], F[:, 7] * 100, "--", label="grain 2")
    ax[1, 1].legend(); ax[1, 1].set(xlabel="E (V/Å)", ylabel="grain strain ∥ (%)", title="grain-resolved strain (fit)")
    fig.suptitle(f"{metrics['run']}  n={np.round(n, 3)}  T={prm.get('T', '?')} K"); fig.tight_layout()
    fig.savefig(out / "quicklook.png", dpi=130)
    print(json.dumps(metrics["global_loop"], indent=1))


if __name__ == "__main__":
    main()
