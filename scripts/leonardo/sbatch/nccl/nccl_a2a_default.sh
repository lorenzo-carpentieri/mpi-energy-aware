#!/bin/bash
#SBATCH --job-name=nccl_a2a_default
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=1	
#SBATCH --time=10:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_NETTUNE_0
#SBATCH --output=/leonardo/home/userexternal/lcarpent/mpi-energy/mpi-energy-aware/logs-leonardo-a2a-default/%x.%j.out
#SBATCH --profile=Energy  # Enables energy profiling
#SBATCH --exclusive


source ./scripts/leonardo/env/set_nccl_env.sh
export NCCL_DEBUG=INFO     # Very detailed logs (verbose)
export NCCL_DEBUG_SUBSYS=GRAPH,COLL,INIT

### CUDA RUN ###
# a2a
mpirun  ./cuda/exe/a2a_nccl /leonardo/home/userexternal/lcarpent/mpi-energy/mpi-energy-aware/logs-leonardo-a2a-default/power/a2a_default /leonardo/home/userexternal/lcarpent/mpi-energy/mpi-energy-aware/logs-leonardo-a2a-default/perf/a2a_default.csv
