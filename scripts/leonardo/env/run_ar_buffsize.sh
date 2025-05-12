#!/bin/bash
#SBATCH --job-name=all_reduce_buf_size
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --time=00:10:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_NETTUNE_0
#SBATCH --output=/leonardo/home/userexternal/lcarpent/mpi-energy/mpi-energy-aware/logs/perf/nccl/ar/buf_size/%x.%j.out
#SBATCH --profile=Energy  # Enables energy profiling
#SBATCH --exclusive


module load cuda openmpi/ nvhpc/ nccl/ 
mkdir -p /leonardo/home/userexternal/lcarpent/mpi-energy/mpi-energy-aware/logs/power/nccl/ar/buf_size/

# Loop over each value
buf_size=$1
export NCCL_BUFFSIZE=$buf_size


echo "Running with NCHANNELS $1"
### CUDA RUN ###
# all_reduce
mpirun  ./cuda/exe/ar_nccl /leonardo/home/userexternal/lcarpent/mpi-energy/mpi-energy-aware/logs/power/nccl/ar/buf_size/ar_${buf_size}
