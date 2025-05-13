#!/bin/bash
#SBATCH --job-name=all_reduce_buffsize
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=1	
#SBATCH --time=10:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_NETTUNE_0
#SBATCH --output=/leonardo/home/userexternal/lcarpent/mpi-energy/mpi-energy-aware/logs/perf/nccl/ar/buffsize/%x.%j.out
#SBATCH --profile=Energy  # Enables energy profiling
#SBATCH --exclusive


module load cuda openmpi/ nvhpc/ nccl/ 

# Loop over each value
buf_size=$1
export NCCL_BUFFSIZE=$buf_size


echo "Running with NCCL_BUFFSIZE $1"
### CUDA RUN ###
# all_reduce
mpirun  ./cuda/exe/ar_nccl /leonardo/home/userexternal/lcarpent/mpi-energy/mpi-energy-aware/logs/power/nccl/ar/buffsize/ar_${buf_size}
