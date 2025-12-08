#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=xxxxxxx
#SBATCH --qos=low
#SBATCH -J namd_cas3o2e_h3ion_50trajs
#SBATCH --ntasks=50
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00

module load anaconda3/2024.10.1 mpi/latest mkl/latest
source /xxxxxxx/.bashrc
conda activate namd

export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so

srun bash -c "source /xxxxxxx/.bashrc && conda activate namd && /xxxxxxx/.conda/envs/namd/bin/python -u /xxxxxxxxx/namd_0815/namd_h3+_vqe_qseEX_hft_curvatureblock.py