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
    df = pd.read_csv(file_path, names=["timestamp", "power"], skiprows=1)
    df["timestamp_ms"] = df["timestamp"].apply(parse_timestamp)
    df["time_diff"] = df["timestamp_ms"].diff().fillna(0) / 1000  # Convert ms to seconds
    df["energy"] = df["power"] * df["time_diff"]  # Energy (kW * s = kJ)
    total_energy = df["energy"].sum() / 1e+6  # kJ to GJ
    return total_energy

def extract_size_from_filename(filename):
    """Extract file size from filename (assuming it contains 1B, 8B, 1KB, ..., 1GB)."""
    bytes = int(filename.split("_")[2].replace(".pow", "").replace("B", ""))
    return bytes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--power-dir", help="Path to the directory containing the data files")
    parser.add_argument("--output", help="Path to save the CSV file", default="results.csv")
    args = parser.parse_args()
    
    apps = ["ar_baseline", "ar_aware"]
    results = []
    files_and_dirs = sorted(os.listdir(args.power_dir))

    for filename in files_and_dirs:
        file_path = os.path.join(args.power_dir, filename)
        if os.path.isfile(file_path):
            size_bytes = extract_size_from_filename(filename)
            if size_bytes is None:
                print(f"Skipping file {filename}: Unable to determine file size.")
                continue
            
            energy = compute_energy(file_path) # energy in Giga Joul
            efficiency = (size_bytes / 1.25e+8) / energy if energy > 0 else 0
            
            approach = apps[0] if apps[0] in filename else apps[1]
            results.append((approach, size_bytes, energy, efficiency))
            print(f"{filename}: {size_bytes} bytes, {energy:.2f} kJ, {efficiency:.15f} Gb/GJ")
    
    df_results = pd.DataFrame(results, columns=["approach", "num_byte", "energy [GJ]", "Gb/GJ"])
    df_results = df_results.sort_values(by=["approach", "num_byte"], ascending=[True, True])
    df_results.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
