#!/bin/bash
#SBATCH --job-name=mpi_energy_tuning
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=1	
#SBATCH --time=10:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_NETTUNE_0
#SBATCH --output=/leonardo/home/userexternal/lcarpent/mpi-energy/mpi-energy-aware/logs/perf/nccl/allGather/%x.%j.out
#SBATCH --profile=Energy  # Enables energy profiling
#SBATCH --exclusive


source ./scripts/leonardo/env/set_nccl_env.sh
export NCCL_DEBUG=TRACE     # Very detailed logs (verbose)

# Loop over each value
alg=$1
prot=$2
nthreads=$3
nchannels=$4
# TODO: add other env variables like buff size and launch mode, 

# Parameters are tuned by NCCL. 
export NCCL_ALGO=$alg
export NCCL_PROTO=$prot
export NCCL_NTHREADS=$nthreads
export NCCL_MIN_CTAS=$nchannels
export NCCL_MAX_CTAS=$nchannels

echo "Running with NCCL_PROTO  $prot, NCCL_ALGO $alg, NCCL_NTHREADS $nthreads, NCCL_MAX_CTAS $nchannels"
### CUDA RUN ###
# allGather
mpirun  ./cuda/exe/allGather_nccl /leonardo/home/userexternal/lcarpent/mpi-energy/mpi-energy-aware/logs/power/nccl/allGather/allGather_prot${prot}_alg${alg}_threads${nthreads}_channels${nchannels} /leonardo/home/userexternal/lcarpent/mpi-energy/mpi-energy-aware/logs/perf/nccl/allGather/allGather_prot${prot}_alg${alg}_threads${nthreads}_channels${nchannels}.csv
