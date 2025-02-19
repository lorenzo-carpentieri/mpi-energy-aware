#include <sycl/sycl.hpp>
#include <mpi.h>
#include <chrono>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include "power_profiler.hpp"
#include <iomanip> 

#define MAX_RUN 5
#define WARM_UP_RUN 5
#define TIME_TO_ACHIEVE_MS 5000

// Define the size of the data
#define dtype u_int8_t
#define MAX_BUF 100
#define BYTE_STEP 8
#define POWER_SAMPLING_RATE_MS 5

int main(int argc, char *argv[]) {
    int rank, size;
    std::string log_path;
    if (argc!=2)
        return -1;
    else{
        // store the path to log directory
        log_path = argv[1];
    }

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

    // Create a SYnergy queue and device for the selected GPU
    synergy::queue queue(selected_device);
    synergy::device sy_dev= queue.get_synergy_device();
    
    // Print rank, total processes, and selected GPU name
    std::cout << "MPI Rank " << rank << "/" << size 
              << " using GPU: " << selected_device.get_info<sycl::info::device::name>() << std::endl;

    // Contains the different size used for testing
    size_t* buff_size_byte = (size_t*) malloc(sizeof(size_t)*MAX_BUF);
    size_t num_byte = 1;
    int i = 0;
    // loop from 1B to 1GiB
    for(i=0; num_byte <= (1ULL << 30); i++){  
        buff_size_byte[i] = num_byte*sizeof(dtype);
        // if(rank==0)
        //     std::cout << "Buf size: " <<  buff_size_byte[i] <<std::endl;
        num_byte*=BYTE_STEP;
    }

    const int num_iters=i;

    // Allocate memory on device
    dtype *d_sendbuf = (dtype *)sycl::malloc_device(buff_size_byte[num_iters - 1],
                                                queue);
    dtype *d_recvbuf = (dtype *)sycl::malloc_device(buff_size_byte[num_iters - 1],
                                                queue);
    // Initialize device buffer (example: set all elements to rank)
    queue.fill<dtype>(d_sendbuf, static_cast<dtype>(rank), buff_size_byte[num_iters - 1]/sizeof(dtype)).wait();
    queue.fill<dtype>(d_recvbuf, static_cast<dtype>(rank), buff_size_byte[num_iters - 1]/sizeof(dtype)).wait();

    // Allocate host memory for the maximum size 
    dtype *h_sendbuf = (dtype *)malloc(buff_size_byte[num_iters-1]);
    dtype *h_recvbuf = (dtype *)malloc(buff_size_byte[num_iters-1]);

    

    // Time for one byte MPI All reduce in microseconds
    // Warm up run
    for (int i = 0; i < WARM_UP_RUN; i++){
        auto start = std::chrono::high_resolution_clock::now();
        
        // Copy data from device to host
        queue.memcpy(h_sendbuf, d_sendbuf, buff_size_byte[0]).wait();
        // Perform MPI_Allreduce on host buffers
        MPI_Allreduce(h_sendbuf, h_recvbuf,( buff_size_byte[0] / sizeof(dtype)), MPI_INT8_T, MPI_SUM, MPI_COMM_WORLD);
        queue.memcpy(d_recvbuf, h_recvbuf, buff_size_byte[0]).wait();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        #ifdef MPI_ENERGY_DEBUG
        std::cout << "Rank " << rank << " MPI_AllReduce time for  "<< buff_size_byte[0] << " bytes: " << time << std::endl;
        #endif
        // Copy result from host to device
    }

   if (rank==0){
        // Results in csv format
        std::cout << "run,byte,mem_cpy_time_ms,time_ms,host_energy_uj,min_goodput_Gbs"<<std::endl;  
   }
    // // Run MPI_All_Reduce for each size
    for(int i = 0; i < num_iters;i++){
        double avg_host_energy_uj=0;
        float avg_time_s=0;
        float avg_mem_cpy_t_s=0;
        // Repeat each test MAX_RUN times
        for(int run = 0; run < MAX_RUN; run++){

            float ar_time = 0;
            double ar_energy_uj=0;
             
            // Initialize device buffer (example: set all elements to rank)
            queue.memset(d_sendbuf, rank, buff_size_byte[i]).wait();

            std::string power_file = log_path + "/ar_baseline_"+ std::to_string(buff_size_byte[i]) + "B.pow";
            PowerProfiler power_prof(sy_dev, POWER_SAMPLING_RATE, power_file);
            // Start GPU power profiling
            power_prof.start();
            // Read host energy 
            auto host_energy_uj_start = queue.host_energy_consumption();
            // Copy data from device to host 
            auto  mem_cpy_t_start = std::chrono::high_resolution_clock::now();
            queue.memcpy(h_sendbuf, d_sendbuf, buff_size_byte[i]).wait();
            auto  mem_cpy_t_end = std::chrono::high_resolution_clock::now();
            // Read host energy 
            auto mem_cpy_host_energy_uj = queue.host_energy_consumption()-host_energy_uj_start;
            int chain_size = 0;
            while((ar_time) < (TIME_TO_ACHIEVE_MS*1000)){  
                auto host_energy_uj_start = queue.host_energy_consumption();
                auto start_s = std::chrono::high_resolution_clock::now();      
                // Perform MPI_Allreduce on host buffers
                MPI_Allreduce(h_sendbuf, h_recvbuf,( buff_size_byte[i] / sizeof(dtype)), MPI_INT8_T, MPI_SUM, MPI_COMM_WORLD);
                auto end_s = std::chrono::high_resolution_clock::now();
                ar_time += std::chrono::duration_cast<std::chrono::microseconds>(end_s - start_s).count();
                auto host_energy_uj_end = queue.host_energy_consumption();
                ar_energy_uj+= host_energy_uj_end - host_energy_uj_start;
                MPI_Allreduce(MPI_IN_PLACE, &ar_time, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
                chain_size++;
            }

            // Copy result from host to device
            queue.memcpy(d_recvbuf, h_recvbuf, buff_size_byte[i]).wait();
            power_prof.stop();

            // for baseline we have to consider also the memcopy times
            float mem_cpy_t =  std::chrono::duration_cast<std::chrono::microseconds>(mem_cpy_t_end - mem_cpy_t_start).count();

            if (rank==0){
                // *2 for host-device and device-host data transfer
                float mem_cpy_t_s = (mem_cpy_t*2) / 1e+6;
                auto ar_single_run_energy_uj = (ar_energy_uj / chain_size)+(mem_cpy_host_energy_uj*2);
                avg_host_energy_uj+=ar_single_run_energy_uj;
                // Data in Giga Byte
                float data_GiB = static_cast<double>(buff_size_byte[i]) / 1e+9;
                // Byte to Giga bit conversion
                float data_Gb = static_cast<double>(buff_size_byte[i]) / 1.25e+8;
                float ar_time_s =( ar_time / 1e+6);
                // Time for a single run of MPI_Allreduce 
                float single_run_time_s = (ar_time_s / chain_size) + mem_cpy_t_s;
                // Average time over all runs
                avg_time_s+=single_run_time_s;
                avg_mem_cpy_t_s+=mem_cpy_t_s;
                // Print results for each run
                std::cout << "run_"<<run << ","<< buff_size_byte[i] << ","<< mem_cpy_t_s * 1000 <<"," << single_run_time_s*1000 << ","<< ar_single_run_energy_uj <<"," << std::fixed << std::setprecision(15) << (data_Gb/single_run_time_s) <<std::endl;  
            }

        }
        
        if(rank==0){
            float data_Gb = static_cast<double>(buff_size_byte[i]) / 1.25e+8;
            avg_time_s/=MAX_RUN;
            avg_host_energy_uj/=MAX_RUN;
            avg_mem_cpy_t_s/=MAX_RUN;
            std::cout << "run_avg" << ","<< buff_size_byte[i] << ","<< avg_mem_cpy_t_s * 1000 <<"," << avg_time_s*1000 << ","<< avg_host_energy_uj <<"," << std::fixed << std::setprecision(15) << (data_Gb/avg_time_s) <<std::endl;  
        }
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
