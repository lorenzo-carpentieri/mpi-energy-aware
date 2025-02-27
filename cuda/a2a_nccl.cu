#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "power-profiler/power_prof.hpp"

#define MAX_RUN 5
#define WARM_UP_RUN 5
#define TIME_TO_ACHIEVE_MS 500
#define POWER_SAMPLING_RATE_MS 5
#define dtype uint8_t
#define MAX_BUF 100
#define BYTE_STEP 8

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
    ncclComm_t comm;
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(&comm, size, id, rank);
    
    size_t *buff_size_byte = (size_t *)malloc(sizeof(size_t) * MAX_BUF);
    size_t num_byte = 1;
    int i = 0;
    for (i = 0; num_byte <= (1ULL << 30); i++) {
        buff_size_byte[i] = num_byte * sizeof(dtype);
        num_byte *= BYTE_STEP;
    }
    const int num_iters = i;

    dtype *d_sendbuf, *d_recvbuf;
    cudaMalloc((void **)&d_sendbuf, buff_size_byte[num_iters - 1]);
    cudaMalloc((void **)&d_recvbuf, buff_size_byte[num_iters - 1]);

    dtype *h_sendbuf = (dtype *)malloc(buff_size_byte[num_iters - 1]);
    dtype *h_recvbuf = (dtype *)malloc(buff_size_byte[num_iters - 1]);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int i = 0; i < WARM_UP_RUN; i++) {
        cudaMemcpy(h_sendbuf, d_sendbuf, buff_size_byte[0], cudaMemcpyDeviceToHost);
        auto start = std::chrono::high_resolution_clock::now();
        ncclGroupStart();
        for (int r=0; r<size; r++) {
            ncclSend(d_sendbuf, (buff_size_byte[i] / sizeof(dtype)), ncclUint8, r, comm, stream);
            ncclRecv(d_recvbuf, (buff_size_byte[i] / sizeof(dtype)), ncclUint8, r, comm, stream);          
        }
        ncclGroupEnd();

        cudaStreamSynchronize(stream);
        auto end = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        cudaMemcpy(d_recvbuf, h_recvbuf, buff_size_byte[0], cudaMemcpyHostToDevice);
    }

    if (rank == 0)
        std::cout << "approach,run,chain_size,byte,mem_cpy_time_ms,time_ms,min_goodput_Gbs" << std::endl;
    cudaMemset(d_sendbuf, rank, buff_size_byte[num_iters-1]);
    auto mem_cpy_t_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_sendbuf, d_sendbuf, buff_size_byte[num_iters-1], cudaMemcpyDeviceToHost);
    auto mem_cpy_t_end = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iters; i++) {
        float avg_time_s = 0;
        int chain_size = 0;

        for (int run = 0; run < MAX_RUN; run++) {
            float ar_time = 0;
            chain_size = 0;

            std::string power_file = log_path + "/a2a_nccl_" + std::to_string(buff_size_byte[i]) + "B"+"_rank"+ std::to_string(rank) + ".pow";
            PowerProfiler powerProf(rank % numGPUs, POWER_SAMPLING_RATE_MS, power_file);
            powerProf.start();
            while (ar_time < (TIME_TO_ACHIEVE_MS * 1000)) {
                auto start_s = std::chrono::high_resolution_clock::now();
                ncclGroupStart();
                for (int r=0; r<size; r++) {
                    ncclSend(d_sendbuf, (buff_size_byte[i] / sizeof(dtype)), ncclUint8, r, comm, stream);
                    ncclRecv(d_recvbuf, (buff_size_byte[i] / sizeof(dtype)), ncclUint8, r, comm, stream);          
                }
                ncclGroupEnd();
                cudaStreamSynchronize(stream);
                auto end_s = std::chrono::high_resolution_clock::now();
                ar_time += std::chrono::duration_cast<std::chrono::microseconds>(end_s - start_s).count();
                MPI_Allreduce(MPI_IN_PLACE, &ar_time, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
                chain_size++;
            }
            powerProf.stop();

            float mem_cpy_t = std::chrono::duration_cast<std::chrono::microseconds>(mem_cpy_t_end - mem_cpy_t_start).count();
            
            if (rank == 0) {
                float mem_cpy_t_s = (mem_cpy_t * 2) / 1e+6;
                float data_Gb = static_cast<double>(buff_size_byte[i]) / 1.25e+8;
                float ar_time_s = (ar_time / 1e+6);
                float single_run_time_s = (ar_time_s / chain_size);
                avg_time_s += single_run_time_s;
                std::cout << "a2a_nccl," << "run_" << run << "," << chain_size << "," << buff_size_byte[i] << "," << mem_cpy_t_s * 1000 << "," << single_run_time_s * 1000 << "," << (data_Gb / single_run_time_s) << std::endl;
            }
        }
        if (rank == 0) {
            float data_Gb = static_cast<double>(buff_size_byte[i]) / 1.25e+8;
            avg_time_s /= MAX_RUN;
            std::cout << "a2a_nccl,run_avg," << chain_size << "," << buff_size_byte[i] << ",N/A," << avg_time_s * 1000 << "," << (data_Gb / avg_time_s) << std::endl;
        }
    }
    cudaMemcpy(d_recvbuf, h_recvbuf, buff_size_byte[num_iters-1], cudaMemcpyHostToDevice);


    ncclCommDestroy(comm);
    cudaStreamDestroy(stream);
    cudaFree(d_sendbuf);
    cudaFree(d_recvbuf);
    free(h_sendbuf);
    free(h_recvbuf);
    MPI_Finalize();

    return 0;
}
