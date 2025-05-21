# x-axis: GB/s
# y-axis: Energy J
# each point is a different NTHREADS 64, 128, 256, 512
# I have a plot with one row and 4 columns each subplot shows the data for a different size.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import itertools
import re
import math
from paretoset import paretoset

all_markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "H", "+", "x", "X", "|", "_"]

tuning_parameters_map={
    "nchannels": "NCCL_MIN/MAX_CTAS",
    "nthreads": "NCCL_NTHREADS",
    "launch": "NCCL_LAUNCH_MODE",
    "buffsize":"NCCL_BUFFSIZE"
}

collectives=["ar"]
# tuning_parameters=["nchannels", "nthreads", "launch", "buffsize"]
# tuning_parameters=["nchannels", "nthreads"] # TODO: use all tuning paramters
tuning_parameters=["launch"] # TODO: use all tuning paramters

libraries=["nccl"]


# from paretoset import paretoset
byte_mapping = {
    1: "1B",
    4: "4B",
    8: "8B",
    16: "16B",
    64: "64B",
    256: "256B",
    512: "512B",
    1024: "1024B",
    4096: "4KiB",
    16384: "16 KiB",
    32768: "32KiB",
    65536: "64 KiB",
    262144: "256KiB",
    1048576:"1024 KiB",
    2097152: "2MiB",
    4194304:"4MiB",
    16777216: "16MiB",
    67108864: "64MiB",
    134217728: "128MiB",
    268435456: "256MiB",
    1073741824: "1GiB"
}



def main():
    parser = argparse.ArgumentParser(description="NCCL energy characterization with different parameters")
    parser.add_argument('--log-dir', type=str, required=True, help="Directory containing the PERFORMAANCE data related to each library collective and tuning parameter. e.g nccl, ar, nthreads")
    parser.add_argument('--out-dir', type=str, required=True, help="path to the plot directory. The script generate in this directory new path. e.g. out-dir/lib/collectives/tuning_parameter/ar_lib_tuning_parameter.pdf")
    parser.add_argument('--app', type=str, required=True, help="Select for which application generate the plot (e.g ar, a2a)")

    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
    app = args.app
 
    all_dfs = []

    # match protocol, algorithm, threads and channels
    pattern = re.compile(r"prot(\w+)_alg(\w+)_threads(\d+)_channels(\d+)\.csv")
    for csv_file_path in log_dir.glob("*.csv"):
        df = pd.read_csv(csv_file_path)
        match = pattern.search(csv_file_path.name)
        if match:
            prot, alg, threads, channels = match.groups()
            
            # Read CSV
            df = pd.read_csv(csv_file_path)
            
            # Add extracted metadata
            df['prot'] = prot
            df['alg'] = alg
            df['threads'] = int(threads)
            df['channels'] = int(channels)
            all_dfs.append(df)
    
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    
    
    os.makedirs(out_dir, exist_ok=True)
    final_df = final_df[final_df["run"]=='run_avg']
    # Ensure both columns have the same type
    final_df["num_byte"] = final_df["num_byte"].astype(int)

    final_df["approach"] = final_df["approach"].astype(str)
    final_df["device_energy"] = final_df["device_energy"].astype(float)
    final_df["host_energy"] = final_df["host_energy"].astype(float)

    # device energy in MJ
    final_df['device_energy [MJ]']= final_df['device_energy'] / 1_000_000
    final_df['GbJ'] = (final_df['num_byte'] / 1.25e+8) / (final_df['device_energy [MJ]']*1e6)
    
   
    final_df = final_df[final_df["approach"].str.contains(app+"_", na=False)] # Extract the data related to the collective specified by app
    # Create a new column for hue and style
    final_df["Protocols x Algorithms"] = final_df["prot"] + " - " + final_df["alg"]
    final_df["Threads x Channels"] = final_df["threads"].astype(str) + " - " + final_df["channels"].astype(str)
    final_df["Host Energy (J)"] = final_df["host_energy"] / 1000 # millijoule to joule
    final_df.to_csv(f"{out_dir}/nccl_{app}_all_params.csv", index=False)


 
if __name__ == "__main__":
    main()


