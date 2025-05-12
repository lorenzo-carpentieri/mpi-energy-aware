#!/bin/bash
#SBATCH --job-name=all_reduce_nthreads
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --time=10:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_NETTUNE_0
#SBATCH --output=/leonardo/home/userexternal/lcarpent/mpi-energy/mpi-energy-aware/logs/perf/nccl/ar/nthreads/%x.%j.out
#SBATCH --profile=Energy  # Enables energy profiling
# module load cuda/12.3
###################################
# export I_MPI_OFFLOAD=1
# export I_MPI_OFFLOAD_MODE=cuda
# export I_MPI_DEBUG=0
####################################

# export I_MPI_OFFLOAD_COLL_PIPELINE=1


module load cuda openmpi/ nvhpc/ nccl/ 
num_threads_per_block=$1
# Loop over each value
export NCCL_NTHREADS=$num_threads_per_block
echo "Running with NCCL_NTHREADS $num_threads_per_block"
### CUDA RUN ###
# mpirun  ./cuda/exe/a2a_cuda_baseline /leonardo/home/userexternal/lcarpent/mpi-energy/mpi-energy-aware/logs/nccl-nthreads/a2a_cuda_baseline_$num_threads_per_block
# nsys profile  --trace=cuda,mpi --mpi-impl=openmpi mpirun  ./cuda/exe/a2a_nccl /leonardo/home/userexternal/lcarpent/mpi-energy/mpi-energy-aware/logs/nccl-nthreads/a2a_nccl_$num_threads_per_block
# mpirun  ./cuda/exe/ar_cuda_baseline /leonardo/home/userexternal/lcarpent/mpi-energy/mpi-energy-aware/logs/nccl-nthreads/ar_cuda_baseline_$num_threads_per_block
mpirun  ./cuda/exe/ar_nccl /leonardo/home/userexternal/lcarpent/mpi-energy/mpi-energy-aware/logs/power/nccl/ar/nthreads/ar_$num_threads_per_block

