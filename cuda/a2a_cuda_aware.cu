#include <mpi.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "power-profiler/power_prof.hpp"

#define MAX_RUN 5
#define WARM_UP_RUN 5
#define TIME_TO_ACHIEVE_MS 5000
#define dtype_small uint8_t
#define dtype_big uint64_t

#define MAX_BUF 100
#define BYTE_STEP 8
#define POWER_SAMPLING_RATE_MS 5

int main(int argc, char *argv[]) {
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

    cudaSetDevice(rank % numGPUs);
   
    size_t *buff_size_byte = (size_t *)malloc(sizeof(size_t) * MAX_BUF);
    size_t num_byte = 1;
    int i = 0;
    for (i = 0; num_byte <= (1ULL << 30); i++) {
        buff_size_byte[i] = num_byte * sizeof(dtype_small);
        num_byte *= BYTE_STEP;
    }
    const int num_iters = i;

    dtype_small *d_sendbuf, *d_recvbuf;
    cudaMalloc((void **)&d_sendbuf, size* buff_size_byte[num_iters - 1]);
    cudaMalloc((void **)&d_recvbuf, size*buff_size_byte[num_iters - 1]);

    dtype_small *h_sendbuf = (dtype_small *)malloc(size*buff_size_byte[num_iters - 1]);
    dtype_small *h_recvbuf = (dtype_small *)malloc(size*buff_size_byte[num_iters - 1]);


    
    for (int i = 0; i < WARM_UP_RUN; i++) {
    
        cudaMemcpy(h_sendbuf, d_sendbuf, buff_size_byte[0], cudaMemcpyDeviceToHost);
        auto start = std::chrono::high_resolution_clock::now();
        MPI_Allreduce(d_sendbuf, d_recvbuf, (buff_size_byte[0] / sizeof(dtype_small)), MPI_UINT8_T, MPI_SUM, MPI_COMM_WORLD);
        auto end = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        cudaMemcpy(d_recvbuf, h_recvbuf, buff_size_byte[0], cudaMemcpyHostToDevice);
    }

    if (rank == 0)
        std::cout << "approach,run,chain_size,byte,mem_cpy_time_ms,time_ms,host_energy_uj,min_goodput_Gbs" << std::endl;
    
    cudaMemset(d_sendbuf, rank, size*buff_size_byte[num_iters-1]);
    cudaMemcpy(h_sendbuf, d_sendbuf, size*buff_size_byte[num_iters-1], cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_iters; i++) {
        double avg_host_energy_uj = 0;
        float avg_time_s = 0;
        float avg_mem_cpy_t_s = 0;
        int chain_size = 0;
        

        for (int run = 0; run < MAX_RUN; run++) {
            double mpi_time_us = 0;
            double ar_energy_uj = 0;

            chain_size = 0;
            int num_el = buff_size_byte[i] / sizeof(dtype_big);
            std::string power_file = log_path + "/a2a_cuda_aware_"+ std::to_string(buff_size_byte[i]) + "B" + "_rank"+ std::to_string(rank) + ".pow";
            PowerProfiler powerProf(rank % numGPUs, POWER_SAMPLING_RATE_MS, power_file);
            powerProf.start();
            while (mpi_time_us < (TIME_TO_ACHIEVE_MS * 1000)) {
                auto start_s = std::chrono::high_resolution_clock::now();
                // if (buff_size_byte[i]>=8UL) // 8 byte
                //     MPI_Alltoall(d_sendbuf, num_el,MPI_UINT64_T, d_recvbuf,  num_el, MPI_UINT64_T, MPI_COMM_WORLD);
                // else
                    MPI_Alltoall(d_sendbuf,(buff_size_byte[i] / sizeof(dtype_small)),MPI_UINT8_T, d_recvbuf, (buff_size_byte[i] / sizeof(dtype_small)), MPI_UINT8_T, MPI_COMM_WORLD);
                auto end_s = std::chrono::high_resolution_clock::now();
                mpi_time_us += std::chrono::duration_cast<std::chrono::microseconds>(end_s - start_s).count();
                MPI_Allreduce(MPI_IN_PLACE, &mpi_time_us, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
              
                chain_size++;
            }
            powerProf.stop();

            
            if (rank == 0) {
                float mem_cpy_t_s = 0;
                float data_Gb = static_cast<double>(buff_size_byte[i]) / 1.25e+8;
                float ar_time_s = (mpi_time_us / 1e+6);
                float single_run_time_s = (ar_time_s / chain_size);
                avg_time_s += single_run_time_s;
                std::cout << "a2a_cuda_aware," << "run_" << run << "," << chain_size << "," << buff_size_byte[i] << "," << mem_cpy_t_s * 1000 << "," << single_run_time_s * 1000 << ",N/A," << (data_Gb / single_run_time_s) << std::endl;
            }
        }
        if (rank == 0) {
            float data_Gb = static_cast<double>(buff_size_byte[i]) / 1.25e+8;
            avg_time_s /= MAX_RUN;
            std::cout << "a2a_cuda_aware,run_avg," << chain_size << "," << buff_size_byte[i] << ",N/A," << avg_time_s * 1000 << ",N/A," << (data_Gb / avg_time_s) << std::endl;
        }
    }

    cudaMemcpy(d_recvbuf, h_recvbuf, buff_size_byte[num_iters-1], cudaMemcpyHostToDevice);

    cudaFree(d_sendbuf);
    cudaFree(d_recvbuf);
    free(h_sendbuf);
    free(h_recvbuf);
    MPI_Finalize();

    return 0;
}
