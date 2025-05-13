#!/bin/bash
#SBATCH --job-name=all_reduce_launch
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1	
#SBATCH --gpus=4
#SBATCH --time=10:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_NETTUNE_0
#SBATCH --output=/leonardo/home/userexternal/lcarpent/mpi-energy/mpi-energy-aware/logs/perf/nccl/ar/launch/%x.%j.out
#SBATCH --profile=Energy  # Enables energy profiling
#SBATCH --exclusive


module load cuda openmpi/ nvhpc/ nccl/ 

# Launch mode can be GROUP or PARALLEL
launch_mode=$1
export NCCL_LAUNCH_MODE=$launch_mode


echo "Running with NCCL_LAUNCH_MODE $launch_mode"
### CUDA RUN ###
# all_reduce
mpirun  ./cuda/exe/ar_nccl /leonardo/home/userexternal/lcarpent/mpi-energy/mpi-energy-aware/logs/power/nccl/ar/launch/ar_${launch_mode}
