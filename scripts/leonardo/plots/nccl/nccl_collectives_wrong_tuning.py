import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import itertools

tuning_parameters_map={
    "nchannels": "NCCL_MIN/MAX_CTAS",
    "nthreads": "NCCL_NTHREADS",
    "launch": "NCCL_LAUNCH_MODE",
    "buffsize":"NCCL_BUFFSIZE"
}
collectives=["ar"]
tuning_parameters=["nchannels", "nthreads", "launch", "buffsize"]
libraries=["nccl"]

all_markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "H", "+", "x", "X", "|", "_"]


byte_mapping = {
    1: "1B",
    8: "8B",
    64: "64B",
    512: "512B",
    4 * 1024: "4KiB",
    32 * 1024: "32KiB",
    256 * 1024: "256KiB",
    16 * 1024 * 1024: "16MiB",
    128 * 1024 * 128: "128MiB",
    1024 * 1024 * 1024: "1GiB"
}

def generate_plot(perf_dir, energy_dir, out_dir, app, obj, lib, coll, tuning_param):
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
    perf_df= pd.read_csv(perf_file_path)
    energy_df= pd.read_csv(energy_file_path)
    ####################################################################################
   
    # Ensure both columns have the same type
    perf_df["num_byte"] = perf_df["num_byte"].astype(int)
    energy_df["num_byte"] = energy_df["num_byte"].astype(int)

    perf_df["approach"] = perf_df["approach"].astype(str)
    energy_df["approach"] = energy_df["approach"].astype(str)
    
    perf_df[tuning_param] = perf_df[tuning_param].astype(int)
    energy_df[tuning_param] = energy_df[tuning_param].astype(int)
    
    # Now merge
    combined_data = pd.merge(perf_df, energy_df[["approach", tuning_param , "num_byte", "energy [MJ]"]],
                            on=["approach", tuning_param, "num_byte"], how="right")
    
    combined_data["chain_size"] = combined_data["chain_size"].astype(float)
    
    combined_data.rename(columns={"energy [MJ]": "device_energy [MJ]"}, inplace=True)
    combined_data['device_energy [MJ]'] /= combined_data['chain_size']
    combined_data['device_energy [MJ]'] *= num_gpus
    combined_data['GbJ'] = (combined_data['num_byte'] / 1.25e+8) / (combined_data['device_energy [MJ]']*1e6) # TODO: consider host time/energy
    combined_data['num_byte'] = combined_data['num_byte'].map(byte_mapping)

    combined_data = combined_data[combined_data["run"]=='run_avg']
    combined_data = combined_data[combined_data["approach"].str.contains(app+"_", na=False)]
    print(combined_data)
    
    ################### Generate marker and palette for each approach ###################################
    # Get unique approaches 
    unique_tuning_params = combined_data[tuning_param].unique()

    # Define available colors and markers
    available_colors = sns.color_palette("tab10", len(unique_tuning_params))  # Get a list of distinct colors
    available_markers = list(itertools.islice(itertools.cycle(all_markers), len(unique_tuning_params)))

    # Create mapping dictionaries
    palette_map = {nthread: color for nthread, color in zip(unique_tuning_params, available_colors)}
    marker_map = {nthread: marker for nthread, marker in zip(unique_tuning_params, available_markers)}
    ######################################### END marker and palette generation ###########################
    print(marker_map)
    
    # Create main plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    if obj=="perf" or  obj=="both":
        sns.pointplot(
            data=combined_data, x="num_byte", y="min_goodput_Gbs", hue=tuning_param, dodge=True, 
            markers=[marker_map[a] for a in unique_tuning_params],  # Map markers
            palette=["tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue"], 
            ax=ax1
        )

        ax1.set_xlabel("")
        ax1.set_ylabel("Min Goodput (Gb/s)")
        ax1.tick_params(axis="y")
        ax1.legend().remove()
        
    ax2 = ax1.twinx()
    if obj=="energy" or obj=="both":   
        # Secondary y-axis for GbJ
        sns.pointplot(
            data=combined_data, x="num_byte", y="GbJ", hue=tuning_param, dodge=True,
            markers=[marker_map[a] for a in unique_tuning_params],  # Map markers
            palette=["tab:green", "tab:green","tab:green","tab:green", "tab:green"], linestyle="--", ax=ax2, legend=False
        )
        ax2.set_ylabel("Gb/J")
        ax2.tick_params(axis="y")
        ax2.legend().remove()
        
    # TODO: the legend depends on the tuning params
    ################ Create legend map ####################
    legend_map = {}
    keywords=["64", "128", "256", "512"] #TODO:  change this value accroding to the tuning params
    for keyword in keywords:
        legend_map[keyword] =int(keyword)
    ######################## END legend map ##################
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker=marker_map[legend_map['64']], color='w', markerfacecolor="tab:blue", markersize=8, label='NTHREAD=64 [Gb/s]'),
        Line2D([0], [0],marker=marker_map[legend_map['64']], color='w', markerfacecolor='tab:green', markersize=8, label='NTHREAD=64 [Gb/J]'),
        Line2D([0], [0],marker=marker_map[legend_map['128']], color='w', markerfacecolor='tab:blue', markersize=8, label='NTHREAD=128 [Gb/s]'),
        Line2D([0], [0], marker=marker_map[legend_map['128']], color='w', markerfacecolor='tab:green', markersize=8, label='NTHREAD=128 [Gb/J]'),
        Line2D([0], [0], marker=marker_map[legend_map['256']], color='w', markerfacecolor='tab:blue', markersize=8, label='NTHREAD=256 [Gb/s]'),
        Line2D([0], [0], marker=marker_map[legend_map['256']], color='w', markerfacecolor='tab:green', markersize=8, label='NTHREAD=256 [Gb/J]'),
        Line2D([0], [0], marker=marker_map[legend_map['512']], color='w', markerfacecolor='tab:blue', markersize=8, label='NTHREAD=512 [Gb/s]'),
        Line2D([0], [0], marker=marker_map[legend_map['512']], color='w', markerfacecolor='tab:green', markersize=8, label='NTHREAD=512 [Gb/J]')
    ]
    ax1.legend(handles=legend_elements, title="")
    
    # plt.title("Min Goodput vs. GbJ Across Different Byte Sizes")
    
    # Create zoomed-in inset plot for the first 4 points
    # Create zoomed-in inset plot for specific num_byte sizes
    zoom_data = combined_data[combined_data["num_byte"].isin(["1B", "8B", "64B", "512B", "4KiB", "32KiB", "256KiB"])]
    zoom_data['time_us']=zoom_data["time_ms"]*1000
    zoom_data['device_energy [J]']=zoom_data["device_energy [MJ]"]*1e6 # energy in joule
    
    axins = inset_axes(ax1, width="45%", height="55%", loc="center left", bbox_to_anchor=(0.09, -0.05, 1.2, 0.8), bbox_transform=ax1.transAxes)  # Move to left center
    if obj=="perf" or  obj=="both":
        sns.pointplot(
            data=zoom_data, x="num_byte", y="time_ms", hue=tuning_param, dodge=True, 
            markers=[marker_map[a] for a in unique_tuning_params],  # Map markers
            palette=["tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue"], ax=axins, legend=False
        )

    axins2 = axins.twinx()
    if obj=="energy" or  obj=="both":
        sns.pointplot(
            data=zoom_data, x="num_byte", y="device_energy [J]", hue=tuning_param, dodge=True,
            markers=[marker_map[a] for a in unique_tuning_params],  # Map markers
            palette=["tab:green", "tab:green", "tab:green", "tab:green", "tab:green"], linestyle="--", ax=axins2, legend=False
        )

    axins.set_xlabel("")
    axins.set_ylabel("Time (ms)", fontsize=10)
    if obj=="perf" or obj=="both":   
        axins.legend_.remove()

    axins2.set_ylabel("Energy (J)", fontsize=10)
    axins.tick_params(axis="both", which="both", labelsize=8)
    axins2.tick_params(axis="both", which="both", labelsize=8)

    # Adjust zoomed-in x limits based on data
    axins.set_xlim(-0.1, 6.5)  # Ensuring the first 4 num_byte sizes are in view

    # Indicate zoom area on main plot
    mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="black", lw=1)
    if obj=="energy" or obj=="both":   
        axins2.legend_.remove()

    plt.savefig(f"{app}.pdf")

def main():
    parser = argparse.ArgumentParser(description="Generate plot comparing Baseline vs GPU-Aware approaches")
    parser.add_argument('--in-perf-dir', type=str, required=True, help="Directory containing the PERFORMAANCE data related to each library collective and tuning parameter. e.g nccl, ar, nthreads")
    parser.add_argument('--in-energy-dir', type=str, required=True, help="Directory containing the ENERGY data related to each library collective and tuning parameter. e.g nccl, ar, nthreads")
    parser.add_argument('--out-dir', type=str, required=True, help="path to the plot directory. The script generate in this directory new path. e.g. out-dir/lib/collectives/tuning_parameter/ar_lib_tuning_parameter.pdf")
    parser.add_argument('--app', type=str, required=True, help="Select for which application generate the plot (e.g ar, a2a)")
    parser.add_argument('--obj', type=str, required=True, help="Select the objectives to plot (e.g perf, energy, both)")
    
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
                generate_plot(tuning_perf_path ,tuning_energy_path, out_dir_tuning, args.app, args.obj, lib, coll, tuning_param)

if __name__ == "__main__":
    main()
