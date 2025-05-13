#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <filesystem>


constexpr auto POWERCAP_ROOT_DIR = "/sys/class/powercap";
constexpr auto POWERCAP_ENERGY_FILE = "energy_uj";
constexpr auto POWERCAP_UNCORE_NAME = "dram";
constexpr auto POWERCAP_CORE_NAME = "core";
constexpr auto POWERCAP_PACKAGE_NAME = "package";


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
        
        #ifdef NVML
        // Capture the total energy at the start
        nvmlReturn_t result = nvmlDeviceGetTotalEnergyConsumption(dev_handle, &start_energy);
        if (result != NVML_SUCCESS) {
            std::cerr << "Error getting start energy: " << nvmlErrorString(result) << std::endl;
        }
        #endif
        start_host_energy = get_powercap_energy();
        profiler_thread = std::thread(&PowerProfiler::profileLoop, this);
        
    }
    
    float stop() {
        running = false;
        double energy_consumed_joules = 0;
       
        if (profiler_thread.joinable()) {
            profiler_thread.join();
        }

        #ifdef NVML
        // Capture the total energy at the start
        nvmlReturn_t result = nvmlDeviceGetTotalEnergyConsumption(dev_handle, &end_energy);
        if (result != NVML_SUCCESS) {
            std::cerr << "Error getting start energy: " << nvmlErrorString(result) << std::endl;
        }
        energy_consumed_joules = static_cast<double>(end_energy - start_energy) / 1e3; // Convert from microjoules to milli joules
        #endif
        end_host_energy = get_powercap_energy();

        return energy_consumed_joules;
    }

    double get_device_energy(){
        return static_cast<double>(end_energy - start_energy) / 1e3;
    }

    double get_host_energy(){
        return static_cast<double>(end_host_energy - start_host_energy) / 1e3;
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
    unsigned long long start_energy;  // Total energy at the start in microjoules
    unsigned long long end_energy;
    double start_host_energy;
    double end_host_energy;


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
        nvmlDeviceGetPowerUsage(dev_handle, &power); // return milliwatts
        return power / 1000; // convert mW to Watt
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

            
    /**
     * @brief Get the names of the host's packages
     * @param base_path The base path of the Powercap interface
     * @return A vector containing the names
     */
    inline std::vector<std::string> get_packages(std::string base_path = POWERCAP_ROOT_DIR) {
        std::vector<std::string> packages;
        std::filesystem::path path(base_path);

        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            if (entry.is_directory()) {
                std::string name = entry.path().filename().string();
            if (name.find(":") == name.length() - 2) {
                packages.push_back(name);
            }
            }
        }
        return packages;
    }


    template <typename... Args>
    std::string build_path(Args... args) {
        std::string path;
        // ((path += "/" + std::string(args)) + ...); // fold expression to concatenate the strings
        ((path += "/" + std::string(args)), ...); // correct fold with comma operator
        return path;
    }

    /**
     * @brief Get the energy consumption of the host in microjoules
     * @details Get the energy consumption of the host in microjoules using the Powercap interface.
     * @return A monotonically increasing value representing the energy consumption of the host in
     * microjoules
     * @throws std::runtime_error if the energy file(s) cannot be opened
    */
    double get_powercap_energy() {
        double energy = 0;
        unsigned long long e;

        // check_root_privileges();

        // if it's a multi-cpu architecture, we want to sum the energy of all the cpus
        for (const auto& p : get_packages()) {
            std::string path = build_path(POWERCAP_ROOT_DIR, p, POWERCAP_ENERGY_FILE);
            std::ifstream file{path, std::ios::in};
            if (!file.is_open()) {
                throw std::runtime_error("host_profiler error: could not open energy register file");
            }
            file >> e;
            energy += static_cast<double>(e);
        }
        return energy;
    }


};
