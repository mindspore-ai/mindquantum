#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=xxxxxxxx
#SBATCH --qos=low
#SBATCH -J xxxxxxxxx
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:40:00

# Load required modules
module load anaconda/3.7.1 mpi/latest mkl/latest

# Activate conda environment
conda activate namd
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment on $(hostname)" >&2
    exit 1
fi

# Set environment variables

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Check if Python is available
which python || { echo "Error: Python not found on $(hostname)" >&2; exit 1; }


# Run the Python script
srun --label /xxxxxxxx/.conda/envs/namd/bin/python -u /xxxxxxxxxxxx/namd_0815/namd_h3+_vqe_qseEX_hft_curvatureblock.py