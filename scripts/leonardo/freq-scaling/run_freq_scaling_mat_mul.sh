#!/bin/bash
#SBATCH --job-name=all_reduce
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod	
#SBATCH --account=IscrC_NETTUNE_0
#SBATCH --output=./freq-scaling-slurm-logs/nccl_ar_freq_scaling_R-%x.%j.out
#SBATCH --gpu-freq=1395
#SBATCH --exclusive

module load cuda openmpi/ nvhpc/ nccl/ 
export CUDA_AUTO_BOOST=0
mkdir /leonardo/home/userexternal/lcarpent/mpi-energy/mpi-energy-aware/logs/ar-freq-scaling/
# Loop over each value
core_freq=$1
nchannels=$2
export NCCL_MIN_NCHANNELS=$nchannels
export NCCL_MAX_NCHANNELS=$nchannels


echo "Running with freq $1 and NCHANNELS $2"
### CUDA RUN ###
mpirun  ./cuda/exe/mat_mul