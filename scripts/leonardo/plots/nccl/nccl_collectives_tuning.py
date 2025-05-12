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
    8: "8B",
    64: "64B",
    512: "512B",
    4096: "4KiB",
    32768: "32KiB",
    262144: "256KiB",
    2097152: "16MiB",
    134217728: "128MiB",
    1073741824: "1GiB"
}


def generate_plot(perf_dir, energy_dir, out_dir, app, lib, coll, tuning_param):
    os.makedirs(out_dir, exist_ok=True)
    
    num_gpus=4 # TODO: Change this if you have more GPUs per node
    
    ##################### Create file path to perf and energy csv #####################
    # There is alway only one file in this direcotry.
    perf_file_path=""
    for perf_file_name in os.listdir(perf_dir):
        perf_file_path = os.path.join(perf_dir, perf_file_name)
    
    energy_file_path=""
    for energy_file_name in os.listdir(energy_dir):
        energy_file_path = os.path.join(energy_dir, energy_file_name)
    ####################################################################################
    
    ################## Generate perf and energy df #####################################################
    print(perf_file_path)
    perf_df= pd.read_csv(perf_file_path)
    energy_df= pd.read_csv(energy_file_path)
    ####################################################################################
    
   
    # Ensure both columns have the same type
    perf_df["num_byte"] = perf_df["num_byte"].astype(int)
    energy_df["num_byte"] = energy_df["num_byte"].astype(int)

    perf_df["approach"] = perf_df["approach"].astype(str)
    energy_df["approach"] = energy_df["approach"].astype(str)
    if "launch" in tuning_param:
        perf_df[tuning_param] = perf_df[tuning_param].astype(str)
        energy_df[tuning_param] = energy_df[tuning_param].astype(str) 
    else:  
        perf_df[tuning_param] = perf_df[tuning_param].astype(int)
        energy_df[tuning_param] = energy_df[tuning_param].astype(int)
        
    # Now merge
    combined_data = pd.merge(perf_df, energy_df[["approach", tuning_param , "num_byte", "energy [MJ]"]],
                            on=["approach", tuning_param, "num_byte"], how="right")
    combined_data["chain_size"] = combined_data["chain_size"].astype(float)
    
    combined_data.rename(columns={"energy [MJ]": "device_energy [MJ]"}, inplace=True)
    combined_data['device_energy [MJ]'] /= combined_data['chain_size']
    combined_data['device_energy [MJ]'] *= num_gpus # I have four GPUs so i should consider the energy consumption of all gpus 
    combined_data['GbJ'] = (combined_data['num_byte'] / 1.25e+8) / (combined_data['device_energy [MJ]']*1e6) # TODO: consider host time/energy

    combined_data = combined_data[combined_data["run"]=='run_avg']
    # combined_data['num_byte'] = combined_data['num_byte'].map(byte_mapping)
    combined_data = combined_data[combined_data["approach"].str.contains(app+"_", na=False)]
    print(combined_data)
    
    ################### Generate marker and palette for each approach ###################################
    # Get unique approaches 
    unique_tuning_params = combined_data[tuning_param].unique()

    # Define available colors and markers
    available_colors = sns.color_palette("tab20", len(unique_tuning_params))  # Get a list of distinct colors
    available_markers = list(itertools.islice(itertools.cycle(all_markers), len(unique_tuning_params)))

    # Create mapping dictionaries
    palette_map = {nthread: color for nthread, color in zip(unique_tuning_params, available_colors)}
    marker_map = {nthread: marker for nthread, marker in zip(unique_tuning_params, available_markers)}
    ######################################### END marker and palette generation ###########################
    print(marker_map)
    
    filter_num_byte=[8, 512, 32768, 262144, 2097152, 134217728, 1073741824]
    
    figureSize=(25,3.5)
    if "buffsize" in tuning_param:
        figureSize=(30,4.5)
   
        
    fig, axes = plt.subplots(1, len(filter_num_byte), figsize=figureSize)  # Adjust figsize as needed
    
    fig.suptitle(f"{lib}/{coll}/{tuning_param}", fontsize=14, fontweight='bold')

    # filter_num_byte=['8B', '512B', '32KiB', '1GiB']
    
    for i, ax in enumerate(axes):
        # take only 8B, 512B, 32KiB, 16MiB, 1GiB
        filtered_data = combined_data[combined_data["num_byte"]==filter_num_byte[i]]
        x_obj="Min Goodput (Gb/s)"
        y_obj="Energy (J)"
        filtered_data.loc[:, y_obj] = filtered_data['device_energy [MJ]'] * 1_000_000  
        filtered_data.loc[:, x_obj] = filtered_data['min_goodput_Gbs']
      
        # For readibility we use different unit measure for each plot
        if i < 3:
            x_obj="Min Goodput Mb/s"
            print(x_obj)
            filtered_data[x_obj] = (filtered_data['min_goodput_Gbs'] * 1_000).round(2)

        # filtered_data[tuning_param] = filtered_data[tuning_param].astype(int)
        filtered_data = filtered_data[[tuning_param, x_obj, y_obj]]
        filtered_data = filtered_data.reset_index(drop=True)
        if "launch" in tuning_param:
            filtered_data[tuning_param] = filtered_data[tuning_param].astype(str)
        else:
            filtered_data[tuning_param] = filtered_data[tuning_param].astype(int)

        print(filtered_data)

        sns.scatterplot(
            data=filtered_data, x=x_obj, y=y_obj, 
            hue=tuning_param, 
            palette='tab20',
            ax=ax
        )
        ax.set_title(f"Message size: {byte_mapping[filter_num_byte[i]]}")
        # ax.legend(ncol=2, title=tuning_param)
        # Get current handles and labels from the plot
        if "buffsize" in tuning_param:       
            handles, labels = ax.get_legend_handles_labels()
            label_map = {}

            for val in unique_tuning_params:
                num = int(val)  # Make sure it's a standard Python int
                if num <= 4096:
                    label = f"{num}B"
                elif num <= 524288:
                    label = f"{num // 1024}KiB"
                else:
                    label = f"{round(num / (1024 * 1024))}MiB"
                label_map[int(num)] = label 

            
            # Apply the new labels using list comprehension
            new_labels = [label_map.get(int(label)) for label in labels]
            
            # Recreate the legend with modified labels
            ax.legend(handles, new_labels, ncol=2, title=tuning_param)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/{coll}_{lib}_{tuning_param}.pdf")
    

def main():
    parser = argparse.ArgumentParser(description="Generate plot comparing Baseline vs GPU-Aware approaches")
    parser.add_argument('--in-perf-dir', type=str, required=True, help="Directory containing the PERFORMAANCE data related to each library collective and tuning parameter. e.g nccl, ar, nthreads")
    parser.add_argument('--in-energy-dir', type=str, required=True, help="Directory containing the ENERGY data related to each library collective and tuning parameter. e.g nccl, ar, nthreads")
    parser.add_argument('--out-dir', type=str, required=True, help="path to the plot directory. The script generate in this directory new path. e.g. out-dir/lib/collectives/tuning_parameter/ar_lib_tuning_parameter.pdf")
    parser.add_argument('--app', type=str, required=True, help="Select for which application generate the plot (e.g ar, a2a)")
    
    args = parser.parse_args()
    in_perf_dir = Path(args.in_perf_dir)
    in_energy_dir = Path(args.in_energy_dir)
    
    out_dir = Path(args.out_dir)
    for lib in libraries:
        library_energy_path=Path(os.path.join(in_energy_dir, lib))
        library_perf_path=Path(os.path.join(in_perf_dir, lib))
        out_dir_lib= Path(os.path.join(out_dir, lib))
        for coll in collectives:
            coll_energy_path=Path(os.path.join(library_energy_path, coll))
            coll_perf_path=Path(os.path.join(library_perf_path, coll))
            out_dir_coll=Path(os.path.join(out_dir_lib, coll))
            
            for tuning_param in tuning_parameters:
                tuning_energy_path=Path(os.path.join(coll_energy_path, tuning_param))
                tuning_perf_path=Path(os.path.join(coll_perf_path, tuning_param))
                out_dir_tuning=Path(os.path.join(out_dir_coll, tuning_param))

                generate_plot(tuning_perf_path ,tuning_energy_path, out_dir_tuning, args.app, lib, coll, tuning_param)

if __name__ == "__main__":
    main()


