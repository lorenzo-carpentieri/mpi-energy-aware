import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
# logs/perf/library/collective/tuning_param/slurm_file
tuning_parameters_map={
    "nchannels": "NCCL_MIN/MAX_CTAS",
    "nthreads": "NCCL_NTHREADS",
    "launch": "NCCL_LAUNCH_MODE",
    "buffsize":"NCCL_BUFFSIZE"
}
collectives=["ar"]
tuning_parameters=["nchannels", "nthreads", "launch", "buffsize"]
libraries=["nccl"]




def parse_timestamp(timestamp):
    """Convert timestamp format HH:MM:SS:ms to milliseconds."""
    h, m, s, ms = map(int, timestamp.split(':'))
    return ((h * 3600) + (m * 60) + s) * 1000 + ms

def compute_energy(file_path):
    """Compute the energy consumption in joules for a given file."""
    df = pd.read_csv(file_path, names=["time", "power"], skiprows=1)
    df["timestamp_ms"] = df["time"].apply(parse_timestamp)
    df["time_diff"] = df["timestamp_ms"].diff().fillna(0) / 1000  # Convert ms to seconds
    df["energy"] = df["power"] * df["time_diff"]  # Energy (W * s)= J
    total_energy = df["energy"].sum() / 1e+6  # J to MegaJoule 
    return total_energy

def extract_size_from_filename(filename):
    """Extract file size from filename (assuming it contains 1B, 8B, 1KB, ..., 1GB)."""
    # split 0 is the collective
    # split 1 is the tuning parameter value
    # split 2 is the size in byte
    bytes = int(filename.split("_")[2].replace("B", ""))
    return bytes


def extract_tunining_param_from_filename(filename):
    """Extract the value of the tuning paramter used for running the collective."""
    # split 0 is the collective
    # split 1 is the tuning parameter value
    # split 2 is the size in byte
    tuning_param_val = filename.split("_")[1]
    return tuning_param_val

def generate_config_name(filename):
    config_name=""
    if 'a2a_' in filename:
        config_name+="a2a"
    elif "ar_" in filename:
        config_name+="ar"
        
    if "cuda" in filename:
        config_name+="_cuda"
    else:
        config_name+="_sycl"
        
    if "nccl" in filename:
        config_name+="_nccl"
    elif "aware" in filename:
        config_name+="_aware"
    else:
        config_name+="_baseline"
        
    return config_name




def parse_file(tuning_dir, out_dir, lib, coll, tuning_param):
    results=[]

    for power_file in os.listdir(tuning_dir):
        power_file_path = os.path.join(tuning_dir, power_file)
        rank=int(power_file.split("_")[len(power_file.split('_'))-1].replace(".pow", '').replace('rank',''))
        if rank!=0:
            continue
        size_bytes = extract_size_from_filename(power_file)
        tuning_val = extract_tunining_param_from_filename(power_file)
        
        if size_bytes is None:
            print(f"Skipping file {power_file}: Unable to determine file size.")
            continue
        # Generate the name of the configuration
        programmingModel=""
        if "nccl" in lib:
            programmingModel="cuda"
        else:
            programmingModel="sycl"

        config_name = f'{coll}_{programmingModel}_{lib}'
        generate_config_name(power_file)
        
        energy = compute_energy(power_file_path) # energy in Mega Joul
        efficiency = (size_bytes / 1.25e+8) / energy if energy > 0 else 0 # Gb/MJ
        
        results.append((config_name, tuning_val, size_bytes, energy, efficiency))
        print(f"{config_name}: {size_bytes} bytes, {energy:.2f} MJ, {efficiency:.15f} Gb/MJ")


    return results
   
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", help="Path to the power directory containinng all the power files")
    parser.add_argument("--out-dir", help="Path to output folder wher power file should be stored for each library, collective and tunining paramters", default="results.csv")
    args = parser.parse_args()
    
    # build file name <MPI_coll>_<impl:cuda/sycl>_<approach:nccl/baseline/...>_<bytes>
    apps = ["ar", "a2a"]
    impl=["cuda"]
    approach=['baseline', 'nccl','aware']
    bytes=[] 
    b=1
    for i in range(0,9):
       bytes.append(b)
       b*=8
    
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    results=[]
    
    for lib in libraries:
        library_path=Path(os.path.join(in_dir, lib))
        out_dir_lib= Path(os.path.join(out_dir, lib))
        for coll in collectives:
            coll_path=Path(os.path.join(library_path, coll))
            out_dir_coll=Path(os.path.join(out_dir_lib, coll))
            
            for tuning_param in tuning_parameters:
                tuning_path=Path(os.path.join(coll_path, tuning_param))
                out_dir_tuning=Path(os.path.join(out_dir_coll, tuning_param))
                os.makedirs(out_dir_tuning, exist_ok=True)
                out_file_path=Path(os.path.join(out_dir_tuning, f'{lib}_{coll}_{tuning_param}.csv'))
                results=parse_file(tuning_path,out_dir_tuning, lib, coll, tuning_param)

                df_results = pd.DataFrame(results, columns=["approach",tuning_param, "num_byte", "energy [MJ]", "Gb/MJ"])
                df_results = df_results.sort_values(by=["approach", tuning_param, "num_byte"], ascending=[True, True, True])
                print(out_file_path)
                df_results.to_csv(out_file_path, index=False)


if __name__ == "__main__":
    main()
