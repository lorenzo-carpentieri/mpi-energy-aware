#include <sycl/sycl.hpp>
#include <mpi.h>
#include <chrono>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>

#define WARM_UP_RUN 5
#define TIME_TO_ACHIEVE_MS 2000

// Define the size of the data
#define dtype u_int8_t
#define MAX_BUF 12
#define BYTE_STEP 8



int main(int argc, char *argv[]) {
    int rank, size;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
   

  

    // Get number of available GPUs
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    int numGPUs = devices.size();

    if (numGPUs == 0) {
        std::cerr << "No GPU devices available!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Assign a GPU to each rank in a round-robin fashion
    sycl::device selected_device = devices[rank % numGPUs];

    // Create a SYCL queue for the selected device
    sycl::queue queue(selected_device);
      // Print rank, total processes, and selected GPU name
    std::cout << "MPI Rank " << rank << "/" << size 
              << " using GPU: " << selected_device.get_info<sycl::info::device::name>() << std::endl;

    // Contains the different size used for testing
    size_t* buff_size_byte = (size_t*) malloc(sizeof(size_t)*MAX_BUF);
    size_t num_byte = 1;
    for(int i=0; i < MAX_BUF; i++){  
        buff_size_byte[i] = num_byte*sizeof(dtype);
        if(rank==0)
            std::cout << "Buf size: " <<  buff_size_byte[i] <<std::endl;
        num_byte*=BYTE_STEP;
    }

    dtype *d_sendbuf = (dtype *)sycl::malloc_device(buff_size_byte[MAX_BUF - 1],
                                                queue);
    dtype *d_recvbuf = (dtype *)sycl::malloc_device(buff_size_byte[MAX_BUF - 1],
                                                queue);
    // Initialize device buffer (example: set all elements to rank)
    queue.fill<dtype>(d_sendbuf, static_cast<dtype>(rank), buff_size_byte[MAX_BUF - 1]/sizeof(dtype)).wait();
    queue.fill<dtype>(d_recvbuf, static_cast<dtype>(rank), buff_size_byte[MAX_BUF - 1]/sizeof(dtype)).wait();

    // Allocate host memory
    dtype *h_sendbuf = (dtype *)malloc(buff_size_byte[MAX_BUF-1]);
    dtype *h_recvbuf = (dtype *)malloc(buff_size_byte[MAX_BUF-1]);

    

    // Time for one byte MPI All reduce in microseconds
    // Warm up run
    for (int i = 0; i < WARM_UP_RUN; i++){
        // Copy data from device to host
        queue.memcpy(h_sendbuf, d_sendbuf, buff_size_byte[0]).wait();

        auto start = std::chrono::high_resolution_clock::now();
        // Allocate device memory
        // Perform MPI_Allreduce on device buffers
        MPI_Allreduce(d_sendbuf, d_recvbuf,( buff_size_byte[0] / sizeof(dtype)), MPI_INT8_T, MPI_SUM, MPI_COMM_WORLD);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        #ifdef MPI_ENERGY_DEBUG
        std::cout << "Rank " << rank << " MPI_AllReduce time for  "<< buff_size_byte[0] << " bytes: " << time << std::endl;
        #endif
        // Copy result from host to device
        queue.memcpy(d_recvbuf, h_recvbuf, buff_size_byte[0]).wait();
    }

    
   
    size_t *chain_sizes= (size_t *) malloc(sizeof(size_t) * MAX_BUF);
    // Detect chain size for each size
    // Run MPI_All_Reduce for each size
    for(int i = 0; i < MAX_BUF;i++){
        long long time = 0;
    
        auto start = std::chrono::high_resolution_clock::now();
        
        // Perform MPI_Allreduce on host buffers
        MPI_Allreduce(d_sendbuf, d_recvbuf,( buff_size_byte[i] / sizeof(dtype)), MPI_INT8_T, MPI_SUM, MPI_COMM_WORLD);

        auto end = std::chrono::high_resolution_clock::now();
        
        time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        #ifdef MPI_ENERGY_DEBUG
        std::cout << "Rank " << rank << " MPI_AllReduce time for  "<< buff_size_byte[i] << " bytes: " << time << std::endl;
        #endif
        const int chain_size = static_cast<float>(TIME_TO_ACHIEVE_MS) / ((static_cast<float>(time) / 1000.0f));
        chain_sizes[i] = chain_size;

        // Copy result from host to device
        queue.memcpy(d_recvbuf, h_recvbuf, buff_size_byte[i]).wait();
    }
    
    // We consider the chain size for the minimum goodput (i.e. maximum time)
    MPI_Allreduce(MPI_IN_PLACE, chain_sizes, MAX_BUF, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);

    // // Run MPI_All_Reduce for each size
    for(int i = 0; i < MAX_BUF;i++){
        float time = 0;
        
        const int chain_size = chain_sizes[i]*2;
        if (rank == 0 ) 
            std::cout << "Run with chain size: "<< chain_size <<" buffer size: "<< buff_size_byte[i] << std::endl;   
        // Initialize device buffer (example: set all elements to rank)
        queue.memset(d_sendbuf, rank, buff_size_byte[i]).wait();

        // Copy data from device to host
        queue.memcpy(h_sendbuf, d_sendbuf, buff_size_byte[i]).wait();

        auto start = std::chrono::high_resolution_clock::now();
        
        for(int j = 0; j < chain_size; j++){        
            // Perform MPI_Allreduce on host buffers
            MPI_Allreduce(d_sendbuf, d_recvbuf,( buff_size_byte[i] / sizeof(dtype)), MPI_INT8_T, MPI_SUM, MPI_COMM_WORLD);
        }

        auto end = std::chrono::high_resolution_clock::now();
        
        time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // Consider for each rank the maximum time
        MPI_Allreduce(MPI_IN_PLACE, &time, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        if (rank==0){
            // Data in Giga Byte
            float data_GiB = static_cast<double>(buff_size_byte[i]) / 1e+9;
            // Byte to Giga bit conversion
            float data_Gb = static_cast<double>(buff_size_byte[i]) / 1.25e+8;
            float time_s = time / 1e+6;
            // Time for a single run of MPI_Allreduce 
            float single_run_time_s = time_s / chain_size;
            std::cout << "Byte: "<< buff_size_byte[i] << " time_s: " << single_run_time_s <<  " min_goodput (Gb/s): "<< data_Gb/single_run_time_s <<std::endl;    
        }
        // Copy result from host to device
        queue.memcpy(d_recvbuf, h_recvbuf, buff_size_byte[i]).wait();
    }
    


    // Clean up
    sycl::free(d_sendbuf, queue);
    sycl::free(d_recvbuf, queue);
    free(h_sendbuf);
    free(h_recvbuf);
    // Finalize MPI
    MPI_Finalize();

    return 0;
}
