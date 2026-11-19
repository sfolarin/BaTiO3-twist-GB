#!/usr/bin/env python3
"""
Structure generator for BaTiO3 core-shell models (single crystals and symmetric
twist bicrystals about [010]). Writes LAMMPS data files in the exact format the
inputs expect (atom_style full + Bonds + CS-Info section).

Twist bicrystal (hk) about [010]:
  grain 1 has crystal [h 0 k]  along lab x,  grain 2 has [h 0 -k] along lab x.
  Total misorientation theta = 2*atan(k/h); in-plane period L = sqrt(h^2+k^2)*a.
  The GB normal is lab y. Each grain is a stack of complete (010) layers
  (BaO at y = 0, TiO2 at y = a/2), so the cell is stoichiometric and charge
  neutral with no atom deletion. Periodic boundaries give TWO GBs:
  at y = Ly/2 and at y = 0 (= Ly).

Examples
  python gen_bto.py single  --n 25 25 25 --a 4.0           -o sc_25.data
  python gen_bto.py twist   --hk 2 1 --nrep 2 --ncell 10    -o s5_210.data
  python gen_bto.py twist   --hk 2 1 --nrep 2 --ncell 10 --shift 0.25 0.0 -o s5_210_t.data
  python gen_bto.py twist   --hk 2 1 --nrep 2 --ncell 10 --reference -o s5_210_ref.data
    (--reference: same box, same N, grain-1 orientation everywhere = no GB)
"""
import argparse, json, math, sys
import numpy as np

QCORE = {1: 5.042, 2: 4.616, 3: 0.970}
QSHELL = {4: -2.870, 5: -1.544, 6: -2.718}
MASS = {1: 135.327, 2: 45.867, 3: 13.999, 4: 2.0, 5: 2.0, 6: 2.0}  # same as original data files
# fractional basis within the cubic cell, split into (010) layers
LAYER_BAO = [(1, (0.0, 0.0)), (3, (0.5, 0.5))]                     # y = 0   : Ba (0,0,0), O (1/2,0,1/2)
LAYER_TIO2 = [(2, (0.5, 0.5)), (3, (0.5, 0.0)), (3, (0.0, 0.5))]   # y = a/2 : Ti, O(1/2,1/2,0), O(0,1/2,1/2)


def write_data(fname, ions, box, title):
    """ions: list of (core_type, x, y, z). Shell placed on core."""
    lx, ly, lz = box
    natoms = 2 * len(ions)
    with open(fname, "w") as f:
        f.write(f"{title}\n\n{natoms} atoms\n{len(ions)} bonds\n6 atom types\n3 bond types\n\n")
        f.write(f"0 {lx:.8f} xlo xhi\n0 {ly:.8f} ylo yhi\n0 {lz:.8f} zlo zhi\n\n")
        f.write("Masses\n\n" + "".join(f"{t} {m}\n" for t, m in MASS.items()) + "\n")
        f.write("Atoms # full\n\n")
        aid = 0
        for i, (t, x, y, z) in enumerate(ions):
            mol = i + 1
            f.write(f"{aid+1} {mol} {t} {QCORE[t]} {x:.8f} {y:.8f} {z:.8f}\n")
            f.write(f"{aid+2} {mol} {t+3} {QSHELL[t+3]} {x:.8f} {y:.8f} {z:.8f}\n")
            aid += 2
        f.write("\nBonds\n\n")
        for i, (t, *_ ) in enumerate(ions):
            f.write(f"{i+1} {t} {2*i+1} {2*i+2}\n")
        f.write("\nCS-Info\n\n")
        for i in range(len(ions)):
            f.write(f"{2*i+1} {i+1}\n{2*i+2} {i+1}\n")
    q = sum(QCORE[t] + QSHELL[t + 3] for t, *_ in ions)
    if abs(q) > 1e-6:
        raise RuntimeError(f"cell not neutral: {q}")


def single_crystal(n, a):
    nx, ny, nz = n
    ions = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for t, (fx, fz) in LAYER_BAO:
                    ions.append((t, (i + fx) * a, j * a, (k + fz) * a))
                for t, (fx, fz) in LAYER_TIO2:
                    ions.append((t, (i + fx) * a, (j + 0.5) * a, (k + fz) * a))
    return ions, (nx * a, ny * a, nz * a)


def rotated_layer(h, k, a, sign, L, layer, y, shift=(0.0, 0.0)):
    """Points of one (010) layer of a crystal whose [h 0 sign*k] lies along lab x,
    wrapped into an LxL periodic box. Returns list of (type, x, y, z)."""
    kk = sign * k
    norm = math.hypot(h, kk)
    ex = np.array([h, kk]) / norm          # crystal (x,z) direction that maps to lab x
    ez = np.array([-kk, h]) / norm         # orthogonal in-plane direction -> lab z
    nmax = int(math.ceil(L / a * 1.5)) + 2
    pts = {}
    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            for t, (fx, fz) in layer:
                c = np.array([i + fx, j + fz]) * a
                X = (c @ ex + shift[0]) % L
                Z = (c @ ez + shift[1]) % L
                key = (t, round(X, 4) % round(L, 4), round(Z, 4) % round(L, 4))
                pts[key] = (t, X, y, Z)
    return list(pts.values())


def twist_bicrystal(h, k, a, nrep, ncell, shift=(0.0, 0.0), reference=False, stacking="continuous"):
    """ncell = number of cubic cells (BaO+TiO2 layer pairs) per grain along y.
    stacking='continuous': BaO,TiO2,... runs on through both GBs -> both GBs are TiO2|BaO (same sense);
                           each GB lacks the 2-fold axis that forbids a dipole along the normal, and the two
                           dipoles add -> built-in field across the grains in a periodic cell.
    stacking='reversed'  : TiO2,BaO,... in both grains -> both GBs are BaO|TiO2, i.e. the continuous case with the
                           sense of the termination along y inverted (expect the opposite sign of the normal P).
    stacking='mirror'    : grain 2 has the reversed layer order (TiO2,BaO,...) -> GBs are TiO2|TiO2 and
                           BaO|BaO, each symmetric under the 2-fold axis of the symmetric twist -> no net
                           normal dipole. Each layer is neutral, so the cell stays stoichiometric/neutral."""
    L = math.hypot(h, k) * a * nrep
    ions = []
    for g in (0, 1):
        sign = +1 if (g == 0 or reference) else -1
        sh = (0.0, 0.0) if (g == 0 or reference) else (shift[0] * L / nrep, shift[1] * L / nrep)
        first, second = (LAYER_BAO, LAYER_TIO2)
        if g == 1 and stacking == "mirror" and not reference:
            first, second = (LAYER_TIO2, LAYER_BAO)
        if stacking == "reversed":            # both grains TiO2 first: every GB becomes BaO|TiO2 (opposite sense)
            first, second = (LAYER_TIO2, LAYER_BAO)
        for j in range(ncell):
            y0 = (g * ncell + j) * a
            ions += rotated_layer(h, k, a, sign, L, first, y0, sh)
            ions += rotated_layer(h, k, a, sign, L, second, y0 + 0.5 * a, sh)
    # sanity: each layer pair must hold the right number of formula units
    nfu_expected = round((L / a) ** 2) * 2 * ncell
    counts = {t: sum(1 for x in ions if x[0] == t) for t in (1, 2, 3)}
    if counts[1] != nfu_expected or counts[2] != nfu_expected or counts[3] != 3 * nfu_expected:
        raise RuntimeError(f"stoichiometry error {counts}, expected {nfu_expected} f.u. -- "
                           "check that L is commensurate (nrep, hk)")
    return ions, (L, 2 * ncell * a, L)


# ---- ferroelectric domain walls (R phase, P || <111>) ------------------------------------------
# lab frame: rows = lab x, y, z in crystal coordinates; the wall normal is lab y (as for the GBs).
# Walls are neutral ((P1-P2).n = 0) and mechanically compatible with the R spontaneous strain.
DW_FRAMES = {
    "109": dict(axes=[[1, 0, 0], [0, 0, 1], [0, -1, 0]], per=(1, 1, 1),
                P1=[1, 1, 1], P2=[-1, -1, 1]),                 # 109 deg wall on (001)
    "71":  dict(axes=[[1, -1, 0], [1, 1, 0], [0, 0, 1]], per=(math.sqrt(2), math.sqrt(2), 1),
                P1=[1, 1, 1], P2=[1, 1, -1]),                  # 71 deg wall on (110)
    "180": dict(axes=[[1, 1, 0], [1, -1, 0], [0, 0, -1]], per=(math.sqrt(2), math.sqrt(2), 1),
                P1=[1, 1, 1], P2=[-1, -1, -1]),                # 180 deg wall on (1-10)
}
PEROV = [(1, (0.0, 0.0, 0.0)), (2, (0.5, 0.5, 0.5)),
         (3, (0.5, 0.5, 0.0)), (3, (0.5, 0.0, 0.5)), (3, (0.0, 0.5, 0.5))]


def domain_cell(wall, n, a, delta=0.12, mono=False):
    """Two R domains (P1 for lab y < Ly/2, P2 above) separated by two periodic walls at y = 0 and Ly/2;
    mono=True: P1 everywhere (same box, same N) = reference for the wall energy.
    Seed: Ti displaced +delta*P_hat, O -delta/2*P_hat (the quench finds the actual state)."""
    f = DW_FRAMES[wall]
    R = np.array([np.array(v, float) / np.linalg.norm(v) for v in f["axes"]])    # crystal -> lab
    L = np.array([n[i] * f["per"][i] * a for i in range(3)])
    # all lattice sites whose lab position falls in the box: bound the crystal-index range by the box
    # corners mapped back to crystal coordinates (R is orthonormal, crystal = R^T lab)
    corners = np.array([[x, y, z] for x in (0, L[0]) for y in (0, L[1]) for z in (0, L[2])]) @ R / a
    lo = np.floor(corners.min(0)).astype(int) - 1; hi = np.ceil(corners.max(0)).astype(int) + 1
    ijk = np.stack(np.meshgrid(*[np.arange(lo[i], hi[i] + 1) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    pts = {}
    for t, fr in PEROV:
        lab = ((ijk + np.array(fr)) * a) @ R.T
        keep = np.all(lab > -1e-6, axis=1) & np.all(lab < L + 1e-6, axis=1)
        w = np.mod(lab[keep], L)
        keys = np.round(w, 3) % np.round(L, 3)
        for k, x in zip(map(tuple, keys), w):
            pts[(t,) + k] = (t, x)
    nfu = round(np.prod(L) / a ** 3)
    cnt = {t: sum(1 for v in pts.values() if v[0] == t) for t in (1, 2, 3)}
    if cnt[1] != nfu or cnt[2] != nfu or cnt[3] != 3 * nfu:
        raise RuntimeError(f"stoichiometry error {cnt}, expected {nfu} f.u.")
    P1 = R @ (np.array(f["P1"], float) / math.sqrt(3)); P2 = R @ (np.array(f["P2"], float) / math.sqrt(3))
    ions = []
    for t, w in pts.values():
        Ph = P1 if (mono or w[1] < L[1] / 2 - 1e-6) else P2
        d = {1: 0.0, 2: delta, 3: -delta / 2}[t]
        x = np.mod(w + d * Ph, L)
        ions.append((t, *x))
    ions.sort(key=lambda v: (v[2], v[1], v[3], v[0]))
    return ions, tuple(L), P1.tolist(), P2.tolist()


def sigma_of(h, k):
    s = h * h + k * k
    while s % 2 == 0:
        s //= 2
    return s


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="kind")
    sub.required = True  # (python 3.6 compatible)
    s = sub.add_parser("single"); s.add_argument("--n", type=int, nargs=3, required=True)
    t = sub.add_parser("twist"); t.add_argument("--hk", type=int, nargs=2, required=True)
    t.add_argument("--nrep", type=int, default=1, help="in-plane repeats of the CSL period")
    t.add_argument("--ncell", type=int, required=True, help="cubic cells per grain along y (GB normal)")
    t.add_argument("--shift", type=float, nargs=2, default=(0.0, 0.0),
                   help="rigid in-plane shift of grain 2, in units of the CSL period")
    t.add_argument("--reference", action="store_true", help="no GB: grain-1 orientation everywhere")
    t.add_argument("--seed", type=float, nargs=3, default=None, metavar=("PX", "PY", "PZ"),
                   help="seed polarization: displace Ti by +d and O by -d/2 along this lab vector (A)")
    t.add_argument("--stacking", choices=["continuous", "mirror", "reversed"], default="continuous",
                   help="layer stacking of grain 2 (see twist_bicrystal docstring)")
    w = sub.add_parser("domain"); w.add_argument("--wall", choices=sorted(DW_FRAMES), required=True)
    w.add_argument("--n", type=int, nargs=3, required=True,
                   help="repeats of the lab-frame period along x, y (wall normal), z")
    w.add_argument("--mono", action="store_true", help="single-domain reference (P1 everywhere)")
    w.add_argument("--delta", type=float, default=0.12, help="seed Ti displacement (A)")
    for q in (s, t, w):
        q.add_argument("--a", type=float, default=4.0, help="starting cubic lattice parameter (A)")
        q.add_argument("-o", "--out", required=True)
    args = p.parse_args()

    if args.kind == "single":
        ions, box = single_crystal(args.n, args.a)
        meta = dict(kind="single", n=args.n, a=args.a)
    elif args.kind == "domain":
        ions, box, P1, P2 = domain_cell(args.wall, args.n, args.a, args.delta, args.mono)
        meta = dict(kind="domain_mono" if args.mono else "domain", wall=args.wall, n=args.n, a=args.a,
                    P1_lab=P1, P2_lab=None if args.mono else P2, wall_planes_y=[0.0, box[1] / 2],
                    wall_area_A2=box[0] * box[2])
    else:
        h, k = args.hk
        ions, box = twist_bicrystal(h, k, args.a, args.nrep, args.ncell, tuple(args.shift), args.reference,
                                    args.stacking)
        if args.seed is not None:
            d = np.array(args.seed, float)
            ions = [(t, *(np.array([x, y, z]) + {1: 0.0, 2: 1.0, 3: -0.5}[t] * d)) for t, x, y, z in ions]
        meta = dict(kind="twist_reference" if args.reference else "twist", hk=[h, k],
                    sigma=sigma_of(h, k), theta_total_deg=2 * math.degrees(math.atan2(k, h)),
                    half_rotation_deg=math.degrees(math.atan2(k, h)),
                    nrep=args.nrep, ncell_per_grain=args.ncell, shift=list(args.shift), a=args.a,
                    stacking=args.stacking, seed=args.seed,
                    gb_planes_y=[0.0, box[1] / 2], gb_area_A2=box[0] * box[2])
    meta.update(n_ions=len(ions), n_particles=2 * len(ions), box=list(box))
    write_data(args.out, ions, box, json.dumps(meta))
    json.dump(meta, open(args.out + ".json", "w"), indent=1)
    print(json.dumps(meta))
