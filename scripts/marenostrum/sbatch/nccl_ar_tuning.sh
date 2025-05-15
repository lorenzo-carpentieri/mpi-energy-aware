#!/bin/bash
#SBATCH --job-name=mpi_ar_energy_tuning
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --partition=acc
#SBATCH --qos=acc_debug	
#SBATCH --time=00:01:30
#SBATCH --account=ehpc112
#SBATCH --output=/gpfs/home/unsa/unsa895905/energy-ws/mpi-energy-aware-main/logs/perf/nccl/ar/%x.%j.out
#SBATCH --exclusive


source ./scripts/marenostrum/env/set_nccl_env.sh
export NCCL_DEBUG=TRACE     # Very detailed logs (verbose)

# Loop over each value
alg=$1
prot=$2
nthreads=$3
nchannels=$4

# Parameters are tuned by NCCL. 
export NCCL_ALGO=$alg
export NCCL_PROTO=$prot
export NCCL_NTHREADS=$nthreads
export NCCL_MIN_CTAS=$nchannels
export NCCL_MAX_CTAS=$nchannels

echo "Running with NCCL_PROTO  $prot, NCCL_ALGO $alg, NCCL_NTHREADS $nthreads, NCCL_MAX_CTAS $nchannels"
### CUDA RUN ###
# all_reduce


mpirun  ./cuda/exe/ar_nccl /gpfs/home/unsa/unsa895905/energy-ws/mpi-energy-aware-main/logs/power/nccl/ar/ar_prot${prot}_alg${alg}_threads${nthreads}_channels${nchannels} /gpfs/home/unsa/unsa895905/energy-ws/mpi-energy-aware-main/logs/perf/nccl/ar/ar_prot${prot}_alg${alg}_threads${nthreads}_channels${nchannels}.csv
