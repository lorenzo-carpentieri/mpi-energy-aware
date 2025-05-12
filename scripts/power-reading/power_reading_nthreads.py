import os
import argparse
import pandas as pd
import numpy as np

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
    bytes = int(filename.split("_")[len(filename.split("_"))-2].replace("B", ""))
    return bytes


def extract_nthreads_from_filename(filename):
    """Extract file size from filename (assuming it contains 1B, 8B, 1KB, ..., 1GB)."""
    nthreads = int(filename.split("_")[len(filename.split("_"))-3])
    return nthreads

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
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--power-dir", help="Path to the directory containing the folder related to power log files")
    parser.add_argument("--output", help="Path to save the CSV file", default="results.csv")
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
       
    results=[]
    for power_file in os.listdir(args.power_dir):
        power_file_path = os.path.join(args.power_dir, power_file)
        rank=int(power_file.split("_")[len(power_file.split('_'))-1].replace(".pow", '').replace('rank',''))
        if rank!=0:
            continue
        size_bytes = extract_size_from_filename(power_file)
        nthreads = extract_nthreads_from_filename(power_file)
        
        if size_bytes is None:
            print(f"Skipping file {power_file}: Unable to determine file size.")
            continue
        
        config_name = generate_config_name(power_file)
        
        energy = compute_energy(power_file_path) # energy in Mega Joul
        efficiency = (size_bytes / 1.25e+8) / energy if energy > 0 else 0 # Gb/MJ
        
        results.append((config_name, nthreads, size_bytes, energy, efficiency))
        print(f"{config_name}: {size_bytes} bytes, {energy:.2f} MJ, {efficiency:.15f} Gb/MJ")


    
    df_results = pd.DataFrame(results, columns=["approach","nthreads", "num_byte", "energy [MJ]", "Gb/MJ"])
    df_results = df_results.sort_values(by=["approach", "nthreads", "num_byte"], ascending=[True, True, True])
    df_results.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
