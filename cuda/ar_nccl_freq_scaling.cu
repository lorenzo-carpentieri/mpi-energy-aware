#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "power-profiler/power_prof.hpp"

#define MAX_RUN 15
#define WARM_UP_RUN 5
#define TIME_TO_ACHIEVE_MS 5000
#define POWER_SAMPLING_RATE_MS 5
#define dtype float
#define MAX_BUF 100
#define BYTE_STEP 8

int main(int argc, char *argv[]) {

    nvmlReturn_t result;
    nvmlDevice_t device;
    unsigned int smClock; // Variable to store the core (SM) clock frequency

    // Initialize NVML
    result = nvmlInit();
    if (result != NVML_SUCCESS) {
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
        return 1;
    }

    // Get GPU handle for the first GPU (index 0)
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS) {
        printf("Failed to get device handle: %s\n", nvmlErrorString(result));
        nvmlShutdown();
        return 1;
    }

    float core_freq;
    // Get the current SM (core) clock frequency
    result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &smClock);
    if (result != NVML_SUCCESS) {
        printf("Failed to get SM clock: %s\n", nvmlErrorString(result));
    } else {
        core_freq=smClock;
    }

    // Shutdown NVML
    nvmlShutdown();

    int rank, size;
    std::string log_path;
    if (argc != 2)
        return -1;
    else
        log_path = argv[1];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    if (numGPUs == 0) {
        std::cerr << "No GPU devices available!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    size_t buff_size_byte = 1024 * 1024 * 1024; // 1GB in bytes

    cudaSetDevice(rank % numGPUs);
    ncclComm_t comm;
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(&comm, size, id, rank);
    
  
    dtype *d_sendbuf, *d_recvbuf;
    cudaMalloc((void **)&d_sendbuf, buff_size_byte);
    cudaMalloc((void **)&d_recvbuf, buff_size_byte);

    dtype *h_sendbuf = (dtype *)malloc(buff_size_byte);
    dtype *h_recvbuf = (dtype *)malloc(buff_size_byte);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    
    if (rank == 0)
        std::cout << "approach,run,core_freq,num_byte,mem_cpy_time_ms,time_ms,min_goodput_Gbs,energy_j" << std::endl;
    cudaMemset(d_sendbuf, rank, buff_size_byte);
    auto mem_cpy_t_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_sendbuf, d_sendbuf, buff_size_byte, cudaMemcpyDeviceToHost);
    auto mem_cpy_t_end = std::chrono::high_resolution_clock::now();

    float avg_time_s = 0;
    float mem_cpy_t_s = 0;
    double avg_energy_j=0;
    int chain_size=0;
    for (int run = 0; run < MAX_RUN; run++) {
        float ar_time = 0;
       
        // std::string power_file = log_path + "/ar_nccl_" + std::to_string(buff_size_byte[i]) + "B"+"_rank"+ std::to_string(rank) + ".pow";
        std::string power_file = log_path + "_" + std::to_string(buff_size_byte) + "B"+"_rank"+ std::to_string(rank) + ".pow";
        PowerProfiler powerProf(rank % numGPUs, POWER_SAMPLING_RATE_MS, power_file);
        powerProf.start();
        auto start_s = std::chrono::high_resolution_clock::now();
        // ncclGroupStart();
        // ncclAllReduce(d_sendbuf, d_recvbuf, (buff_size_byte / sizeof(dtype)), ncclUint8, ncclSum, comm, stream);
        // ncclGroupEnd();
        chain_size=0;
        // cudaStreamSynchronize(stream);
        // auto end_s = std::chrono::high_resolution_clock::now();
        // ar_time += std::chrono::duration_cast<std::chrono::microseconds>(end_s - start_s).count();
        // MPI_Allreduce(MPI_IN_PLACE, &ar_time, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        while (ar_time < (TIME_TO_ACHIEVE_MS * 1000)) {
            auto start_s = std::chrono::high_resolution_clock::now();
            ncclGroupStart();
            ncclAllReduce(d_sendbuf, d_recvbuf, (buff_size_byte / sizeof(dtype)), ncclUint8, ncclSum, comm, stream);
            ncclGroupEnd();

            cudaStreamSynchronize(stream);
            auto end_s = std::chrono::high_resolution_clock::now();
            ar_time += std::chrono::duration_cast<std::chrono::microseconds>(end_s - start_s).count();
            MPI_Allreduce(MPI_IN_PLACE, &ar_time, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
            chain_size++;
        }
        double energy=powerProf.stop() / chain_size;
        avg_energy_j+=energy;
        float mem_cpy_t = std::chrono::duration_cast<std::chrono::microseconds>(mem_cpy_t_end - mem_cpy_t_start).count();
        
        if (rank == 0) {
            mem_cpy_t_s = (mem_cpy_t * 2) / 1e+6;
            float data_Gb = static_cast<double>(buff_size_byte) / 1.25e+8;
            float ar_time_s = (ar_time / 1e+6);
            float single_run_time_s = (ar_time_s/chain_size);
            avg_time_s += single_run_time_s;
            std::cout << "ar_cuda_nccl," << "run_" << run  << "," << core_freq << "," << buff_size_byte << "," << mem_cpy_t_s * 1000 << "," << single_run_time_s * 1000 << "," << (data_Gb / single_run_time_s) << "," << energy <<std::endl;
        }
    }
    if (rank == 0) {
        float data_Gb = static_cast<double>(buff_size_byte) / 1.25e+8;
        avg_time_s /= MAX_RUN;
        avg_energy_j /= MAX_RUN;
        std::cout << "ar_cuda_nccl,run_avg," << core_freq << "," << buff_size_byte << ","<< mem_cpy_t_s << "," << avg_time_s * 1000 << "," << (data_Gb / avg_time_s) << "," << avg_energy_j << std::endl;
    }
    cudaMemcpy(d_recvbuf, h_recvbuf, buff_size_byte, cudaMemcpyHostToDevice);
    

    ncclCommDestroy(comm);
    cudaStreamDestroy(stream);
    cudaFree(d_sendbuf);
    cudaFree(d_recvbuf);
    free(h_sendbuf);
    free(h_recvbuf);
    MPI_Finalize();

    return 0;
}
