#!/bin/bash
#SBATCH -A YOUR_NERSC_ACCOUNT
#SBATCH -C cpu
#SBATCH -q {qos}
#SBATCH -N {nodes}
#SBATCH --ntasks-per-node={ntasks}
#SBATCH -c 2
#SBATCH -t {walltime}
#SBATCH -J {run_id}
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL
# Perlmutter CPU node: 128 physical cores (256 hardware threads) -> 128 MPI ranks x 2 threads slots.
# Pure MPI: one OpenMP thread per rank.
export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

# LAMMPS binary: keep it OUTSIDE $PSCRATCH (scratch is purged). Override with: LMP=/path sbatch job.sh
LMP=${{LMP:-$HOME/bin/lmp}}

echo "run {run_id} start $(date) on $SLURM_JOB_NUM_NODES nodes, $SLURM_NTASKS ranks"
echo "$SLURM_JOB_ID" > JOBID
srun --cpu-bind=cores "$LMP" -in in.lammps -log log.lammps {lmp_vars}
status=$?
if grep -q RUN_COMPLETE log.lammps; then echo complete > STATUS; else echo "failed($status)" > STATUS; fi
echo "run {run_id} end $(date) status $(cat STATUS)"
