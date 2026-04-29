#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# MPI + CUDA Genetic Algorithm — AiMOS (POWER9 + V100) SLURM job
#
# Resource args (--nodes, --gres=gpu:N) are passed at sbatch invocation and
# override the defaults below.  The config file is positional arg $1.
#
# Usage:
#   sbatch [--nodes=N] [--gres=gpu:G] run.sbatch <config>
#
# Examples (see full list in sbatch_commands.sh):
#   sbatch --nodes=1 --gres=gpu:1 run.sbatch config/strong_scaling_1gpu.conf
#   sbatch --nodes=2 --gres=gpu:6 run.sbatch config/strong_scaling_2node.conf
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=ga_mpi_cuda
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --nodes=1                  # default — override at submission
#SBATCH --ntasks-per-node=1        # always 1 MPI rank per node
#SBATCH --gres=gpu:1               # default — override at submission
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --partition=dcs-2024
#SBATCH --mem=32G

# ── Environment ──────────────────────────────────────────────────────────────
module purge
module load xl_r spectrum-mpi cuda

CONFIG=${1:-config/default.conf}

echo "=================================================="
echo "Job ID        : $SLURM_JOB_ID"
echo "Nodes         : $SLURM_JOB_NODELIST"
echo "MPI ranks     : $SLURM_NTASKS  (1 per node)"
echo "GPUs per node : $SLURM_GPUS_ON_NODE"
echo "Config        : $CONFIG"
echo "Start         : $(date)"
echo "=================================================="
echo ""

# ── Build ─────────────────────────────────────────────────────────────────────
make -s
if [ $? -ne 0 ]; then
    echo "ERROR: build failed — aborting."
    exit 1
fi

# ── Run ──────────────────────────────────────────────────────────────────────
# spectrum-mpi picks up the SLURM allocation automatically.
# Each rank sees only its node's GPUs; NUM_GPUS=0 in the config uses all of them.
mpirun -np "$SLURM_NTASKS" ./app.o "$CONFIG"

echo ""
echo "Finished : $(date)"
