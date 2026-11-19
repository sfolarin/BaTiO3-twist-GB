#!/bin/bash
# Build every structure used in the paper. Re-running gives byte-identical files.
set -e
cd "$(dirname "$0")/../structures"
mkdir -p data
G="python3 gen_bto.py"
$G single --n 20 20 20 --a 4.0                       -o data/sc_20.data
$G twist  --hk 2 1 --nrep 4 --ncell 10 --a 4.01       -o data/s5_210.data      # = original Sigma5 switching cell (35.87 x 80.2 x 35.87 A, 16000 particles)
$G twist  --hk 6 5 --nrep 2 --ncell 10 --a 4.01       -o data/s61_650.data     # = original Sigma61 cell (62.64 x 80.2 x 62.64 A, 48800 particles)
$G twist  --hk 2 1 --nrep 8 --ncell 20 --a 4.01       -o data/s5_210_L.data    # finite-size: 2x lateral, 2x grain thickness (128000 particles)

# added 2026-10-31: termination test + phase-sequence ramps
$G twist  --hk 2 1 --nrep 4 --ncell 10 --a 4.01 --stacking mirror -o data/s5m_210.data  # mirror stacking: TiO2|TiO2 + BaO|BaO GBs
$G single --n 12 12 12 --a 4.0                        -o data/sc_12.data       # temperature ramps (17280 particles)

# added 2026-11-01: referee check A3 (termination sense reversed: both GBs BaO|TiO2)
$G twist  --hk 2 1 --nrep 4 --ncell 10 --a 4.01 --stacking reversed -o data/s5r_210.data
echo "structures written to structures/data/"
