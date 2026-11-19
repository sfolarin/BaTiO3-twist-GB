#!/bin/bash
# Submit queued job scripts in order, as long as premium slots are free (limit: 5 jobs per user,
# running + pending). Safe to re-run any time; each script is submitted once (marker <script>.submitted).
#   bash tools/submit_free.sh
cd "$(dirname "$0")/.."
LIMIT=5
QUEUE=(runs/_bundles/b10K.sh runs/02_gb_energy/job_gb.sh runs/_bundles/aux0K.sh
       runs/_bundles/b300K_a.sh runs/_bundles/b300K_b.sh)
for j in "${QUEUE[@]}"; do
  [ -f "$j" ] || continue
  [ -f "$j.submitted" ] && continue
  n=$(squeue -u "$USER" -q premium -h | wc -l)
  if [ "$n" -ge "$LIMIT" ]; then echo "premium slots full ($n/$LIMIT); waiting: $j and later"; exit 0; fi
  out=$(cd "$(dirname "$j")" && sbatch "$(basename "$j")") && { echo "$(basename "$j"): $out"; echo "$out" > "$j.submitted"; }
done
echo "nothing left to submit (or all remaining scripts not yet written)"
