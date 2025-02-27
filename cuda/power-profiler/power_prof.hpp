#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <fstream>



#ifdef NVML
#include <nvml.h>
struct nvml {
    using device_identifier = unsigned int;
    using device_handle = nvmlDevice_t;
    using return_type = nvmlReturn_t;
    static constexpr nvmlReturn_t return_success = NVML_SUCCESS;
};
using lib = nvml;
#endif

class PowerProfiler {
public:
    PowerProfiler(lib::device_identifier dev_id, int sampling_rate_ms, std::string file_name) : dev_id(dev_id), sampling_rate(sampling_rate_ms),  log_file(file_name, std::ios::out), running(false) {
        if (!log_file.is_open()) {
            throw std::runtime_error("Failed to open log file: " + file_name);
        }
        initialize_library();
    }
    
    void start() {
        log_file << "time,power"<<std::endl;
        running = true;
        profiler_thread = std::thread(&PowerProfiler::profileLoop, this);
    }
    
    void stop() {
        running = false;
        if (profiler_thread.joinable()) {
            profiler_thread.join();
        }
    }
    
    void printMeasurements() {
        for (const auto& m : measurements) {
            std::cout << "Power: " << m << " KW" << std::endl;
        }
    }
    inline ~PowerProfiler() {
        shutdown_library();
    }
    
private:
    int sampling_rate;
    lib::device_identifier dev_id;
    lib::device_handle dev_handle;

    std::vector<double> measurements;
    std::atomic<bool> running;
    std::ofstream log_file;
    std::thread profiler_thread;


    std::string getCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        std::tm local_time = *std::localtime(&now_c);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        std::ostringstream timestamp;
        timestamp << std::put_time(&local_time, "%H:%M:%S")
                  << ":" << std::setfill('0') << std::setw(3) << ms.count();
        return timestamp.str();
    }

    void initialize_library(){
        #ifdef NVML
            // Inititialize nvml and nvml device
            nvmlInit();
            nvmlDeviceGetHandleByIndex(dev_id, &dev_handle);
        #endif
    }


    void shutdown_library(){
        #ifdef NVML
            nvmlShutdown();
        #endif
    }

    void profileLoop() {

        while (running) {

            std::string timestamp = getCurrentTimestamp();
            double power = readPower();
            measurements.push_back(power);
            log_file << timestamp << "," << power << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(sampling_rate));
        }
    }

    double readPower() {
#ifdef NVML
        return readNvmlPower();
#elif defined(INTEL)
        return readIntelPower();
#elif defined(AMD)
        return readAmdPower();
#else
        return 0.0;
#endif
    }

#ifdef NVML
    double readNvmlPower() {
        unsigned int power;
        nvmlDeviceGetPowerUsage(dev_handle, &power);
        return power / 1000; // return Watt
    }
#endif
    
    double readIntelPower() {
        // Placeholder for Intel power reading implementation
        return 50.0;
    }
    
    double readAmdPower() {
        // Placeholder for AMD power reading implementation
        return 60.0;
    }
};
