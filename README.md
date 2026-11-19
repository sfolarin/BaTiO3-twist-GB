# BaTiO3-twist-GB

Inputs, force fields, analysis scripts and processed data for

> S. Folarin, S. Manna, A. Adekoya, and S. Sankaranarayanan,
> *Termination-controlled polarization pinning and grain-size-dependent remanence at symmetric twist grain
> boundaries in BaTiO3*, submitted to Physical Review Materials (2026).

Molecular dynamics with the anharmonic core–shell potential of Vielma and Schneider
[J. Appl. Phys. 114, 174108 (2013)], run with LAMMPS (CORESHELL and CLASS2 packages).

## Contents

| Folder | What it holds |
|---|---|
| `forcefield/` | `bto_coreshell.ff` (Wolf sum, used for all production runs), `bto_coreshell_pppm.ff` (Ewald/PPPM control runs), `bto_coreshell_A7200.ff` (sensitivity test, see below) |
| `structures/` | `gen_bto.py`: single crystals, symmetric [010] twist bicrystals (continuous, reversed and mirror stacking), domain-wall cells. `data/*.json` record the generator arguments of every structure; `data/s5_210.data` is the Σ5 cell used in Fig. 2 |
| `templates/` | LAMMPS inputs: `hysteresis.in` (field cycles), `tramp.in` (temperature ramps), `gb_energy.in`, `dw_energy.in`, `energy_eval.in` |
| `tools/` | run builders and Slurm helpers: `runs.csv` (one row per hysteresis run), `make_structures.sh`, `make_runs.py`, `make_tramp.py`, `gb_scan.py`, `gb_profile.py`, `dw_scan.py`, `term_test.py`, `phase_bench_lmp.py`, `bundle.py`, `submit_free.sh`, `status.py` |
| `analysis/` | `btoan/` (per-cell charge-neutral polarization, grain strain, loop metrics), `process_run.py`, `process_all.py`, `tramp_analysis.py`, `phase_id.py`, `referee_checks.py`, `referee2.py`, `components.py`, tests |
| `figures/` | `make_figures.py`: every figure and table of the paper and Supplemental Material from the processed data; `out/numbers.json` lists every number quoted in the text |
| `runs/` | processed data of every run: `params.json` (all parameters and SHA-256 of template, force field and structure), `traj.dat`/`traj_T.dat` (field or temperature schedule), `analysis/frames.csv`, `analysis/metrics.json`, `analysis/loop.csv`, `analysis/tramp.csv`, `analysis/phase_id.csv`, and the GB, domain-wall and termination JSON files |

Not included because of their size: trajectories (`*.lammpstrj`), relaxed structures (`*.data`), LAMMPS logs, and
the per-cell polarization files `analysis/cellP.npz`. The `cellP.npz` files are archived on Zenodo (see the paper's
data-availability statement); unpack them over `runs/` to regenerate the figures that use them
[Fig. 4(c,d), part of Fig. 5, and Figs. S2–S4]. Trajectories are available from the corresponding authors on request.

## Force field

The parameters are those of the scaled model of Vielma and Schneider as listed by Boddu, Endres and Steinmann
[Sci. Rep. 7, 806 (2017), Table 1], whose Ti–O Buckingham prefactor is 7220.27 eV; Vielma and Schneider list
7200.27 eV. `bto_coreshell_A7200.ff` uses the latter and was used only for the sensitivity test of SI Sec. S2.
The Buckingham energy is shifted to zero at its cutoff (`pair_modify shift yes`), which leaves all forces unchanged.

## Requirements

* LAMMPS with the CORESHELL and CLASS2 packages (tested with the 29 Aug 2024 release)
* Python ≥ 3.9 with the packages in `requirements.txt`

## Workflow

```bash
bash tools/make_structures.sh                    # structures/data/*.data
python tools/make_runs.py                        # builds runs/<group>/<T>K/<run_id>/ from tools/runs.csv
python tools/bundle.py NAME RUN_ID ...           # optional: several runs in one Slurm job
python tools/status.py                           # state of every run
python analysis/process_all.py                   # analysis/ folder of every finished run
python tools/make_tramp.py                       # temperature ramps -> analysis/tramp_analysis.py, analysis/phase_id.py
python tools/gb_scan.py build | collect          # GB energies (14 boundaries x 16 rigid shifts)
python tools/gb_profile.py build | collect       # layer-resolved GB energies, gamma_loc
python tools/dw_scan.py build | collect          # domain-wall energies
python tools/term_test.py build | collect        # 0 K termination / electrostatics tests
python tools/phase_bench_lmp.py build | run | collect   # 0 K phase energies
python figures/make_figures.py                   # all figures and tables -> figures/out/
```

The Slurm scripts were written for NERSC Perlmutter (CPU nodes). Set your own account and e-mail in
`tools/submit_perlmutter.sh`, and the LAMMPS binary with `export LMP=/path/to/lmp`.

## Figures and the data behind them

| Figure | Data | Script |
|---|---|---|
| Fig. 1 | `runs/01_benchmark/` (0 K phases, `tramp_heat`, `tramp_cool`) | `phase_bench_lmp.py`, `tramp_analysis.py`, `phase_id.py` |
| Fig. 2 | `runs/02_gb_energy/gb_energies.json`, `gb_profiles.json` | `gb_scan.py`, `gb_profile.py` |
| Fig. 3 | `runs/03_single_crystal/` | `process_run.py` |
| Fig. 4 | `runs/04_sigma5/10K/` | `process_run.py` |
| Fig. 5 | `runs/04_sigma5/300K/`, `runs/05_sigma61/`, `runs/06_finite_size/` | `process_run.py` |
| Fig. 6 | `runs/04_sigma5/`, `runs/03_single_crystal/` | `process_run.py` |
| Figs. S1–S5, Tables S1–S9 | as above, plus `runs/07_domain_walls/`, `runs/08_termination/` | `referee_checks.py`, `referee2.py`, `dw_scan.py`, `term_test.py` |

## License

Code: MIT (see `LICENSE`). Data: CC BY 4.0.

## Citation

If you use this repository, please cite the paper above.
