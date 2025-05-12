#!/bin/bash
#SBATCH --job-name=all_reduce
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --time=00:20:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_NETTUNE_0
#SBATCH --output=./freq-scaling-slurm-logs/nccl_ar_freq_scaling_R-%x.%j.out
#SBATCH --profile=Energy  # Enables energy profiling
#SBATCH --gpu-freq=1260
#SBATCH --exclusive


module load cuda openmpi/ nvhpc/ nccl/ 
mkdir /leonardo/home/userexternal/lcarpent/mpi-energy/mpi-energy-aware/logs/ar-freq-scaling/
# Loop over each value
core_freq=$1
nchannels=$2
export NCCL_MIN_NCHANNELS=$nchannels
export NCCL_MAX_NCHANNELS=$nchannels


echo "Running with freq $1 and NCHANNELS $2"
### CUDA RUN ###
mpirun  ./cuda/exe/ar_nccl_freq_scaling /leonardo/home/userexternal/lcarpent/mpi-energy/mpi-energy-aware/logs/ar-freq-scaling/ar_nccl_${core_freq}_${nchannels}
