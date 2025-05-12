#!/bin/bash
#SBATCH --job-name=all_reduce_nchannels
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --time=00:10:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_NETTUNE_0
#SBATCH --output=/leonardo/home/userexternal/lcarpent/mpi-energy/mpi-energy-aware/logs/perf/nccl/ar/nchannels/
#SBATCH --profile=Energy  # Enables energy profiling
#SBATCH --exclusive


module load cuda openmpi/ nvhpc/ nccl/ 

# Loop over each value
nchannels=$1
export NCCL_MIN_CTAS=$nchannels
export NCCL_MAX_CTAS=$nchannels


echo "Running with NCHANNELS $1"
### CUDA RUN ###
# all_reduce
mpirun  ./cuda/exe/ar_nccl /leonardo/home/userexternal/lcarpent/mpi-energy/mpi-energy-aware/logs/power/nccl/ar/nchannels/ar_${nchannels}
