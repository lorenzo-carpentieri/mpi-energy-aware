# x-axis: GB/s
# y-axis: Energy J
# each point is a different NTHREADS 64, 128, 256, 512
# I have a plot with one row and 4 columns each subplot shows the data for a different size.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
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



# def print_pareto(df, color, name):
#     # Compute the pareto set point and print on the plot
#     # Creaete a data frame with energy and speedup
#     df_speedup_energy_accurate = pd.DataFrame({"speedup": df['speedup'], "energy": df['normalized_energy']})
#     mask = paretoset(df_speedup_energy_accurate, sense=["max", "min"])
#     pset = df_speedup_energy_accurate[mask]
    
#     pset = pset.sort_values(by=["speedup"])
    
#     np_array = pset.to_numpy()
#     pset_size = len(pset["speedup"])
    
#     cur_xlim_left, cur_xlim_right = plt.xlim()
#     cur_xlim_bottom, cur_ylim_top = plt.ylim()
#     x1, y1 = [cur_xlim_left, np_array[0][0]], [np_array[0][1], np_array[0][1]]
#     plt.plot(x1, y1, color=color, linewidth=2.5, label=name)

#     for i in range(pset_size):
#         if not (i == pset_size-1):
#             current_x = np_array[i][0]
#             current_y = np_array[i][1]
#             next_x = np_array[i+1][0]
#             next_y = np_array[i+1][1]
#             x1, y1 = [current_x, current_x], [current_y, next_y]
#             x2, y2 = [current_x, next_x], [next_y, next_y]
#             plt.plot(x1, y1, x2, y2, color=color, linewidth=2.5)


def generate_plot(csv_dir, app, obj):
    num_gpus=4
    cuda_time_file = os.path.join(csv_dir, 'nccl_all_reduce_nthreads_time.csv')
    cuda_power_file = os.path.join(csv_dir, 'nccl_all_reduce_nthreads_power.csv')
    
    cuda_time_df= pd.read_csv(cuda_time_file)
    cuda_power_df= pd.read_csv(cuda_power_file)
    
   
    # Ensure both columns have the same type
    cuda_time_df["num_byte"] = cuda_time_df["num_byte"].astype(int)
    cuda_power_df["num_byte"] = cuda_power_df["num_byte"].astype(int)

    cuda_time_df["approach"] = cuda_time_df["approach"].astype(str)
    cuda_power_df["approach"] = cuda_power_df["approach"].astype(str)
    
    cuda_time_df["nthreads"] = cuda_time_df["nthreads"].astype(int)
    cuda_power_df["nthreads"] = cuda_power_df["nthreads"].astype(int)
    
    # Now merge
    combined_data = pd.merge(cuda_time_df, cuda_power_df[["approach", "nthreads" , "num_byte", "energy [MJ]"]],
                            on=["approach", "nthreads", "num_byte"], how="right")
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
    unique_nthreads = combined_data["nthreads"].unique()

    # Define available colors and markers
    available_colors = sns.color_palette("tab10", len(unique_nthreads))  # Get a list of distinct colors
    available_markers = ["o", "s", "^", "D"]  # Expand if needed

    # Create mapping dictionaries
    palette_map = {nthread: color for nthread, color in zip(unique_nthreads, available_colors)}
    marker_map = {nthread: marker for nthread, marker in zip(unique_nthreads, available_markers)}
    ######################################### END marker and palette generation ###########################
    print(marker_map)
    
    fig, axes = plt.subplots(1, 6, figsize=(25, 3.5))  # Adjust figsize as needed
    fig.suptitle("NCCL AllReduce", fontsize=14, fontweight='bold')

    # filter_num_byte=['8B', '512B', '32KiB', '1GiB']
    filter_num_byte=[8, 512, 32768, 262144, 2097152, 1073741824]
    
    for i, ax in enumerate(axes):
        # take only 8B, 512B, 32KiB, 16MiB, 1GiB
        filtered_data = combined_data[combined_data["num_byte"]==filter_num_byte[i]]
        x_obj="Min Goodput (Gb/s)"
        y_obj="Energy (J)"
        filtered_data.loc[:, y_obj] = filtered_data['device_energy [MJ]'] * 1_000_000  
        filtered_data.loc[:, x_obj] = filtered_data['min_goodput_Gbs']
        
        if i < 3:
            x_obj="Min Goodput Mb/s"
            print(x_obj)
            filtered_data[x_obj] = (filtered_data['min_goodput_Gbs'] * 1_000).round(3)

        # filtered_data["nthreads"] = filtered_data["nthreads"].astype(int)
        filtered_data = filtered_data[['nthreads', x_obj, y_obj]]
        filtered_data = filtered_data.reset_index(drop=True)
        filtered_data["nthreads"] = filtered_data["nthreads"].astype(int)

        print(filtered_data)

        sns.scatterplot(
            data=filtered_data, x=x_obj, y=y_obj, 
            hue="nthreads", 
            palette='tab10',
            ax=ax
        )
        ax.set_title(f"Message size: {byte_mapping[filter_num_byte[i]]}")
    plt.tight_layout()
    plt.savefig(f"{app}_nthreads.pdf")
    
    # # Create main plot
    # fig, ax1 = plt.subplots(figsize=(10, 6))
    # if obj=="perf" or  obj=="both":
    #     sns.pointplot(
    #         data=combined_data, x="num_byte", y="min_goodput_Gbs", hue="nthreads", dodge=True, 
    #         markers=[marker_map[a] for a in unique_nthreads],  # Map markers
    #         palette=["tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue"], 
    #         ax=ax1
    #     )

    #     ax1.set_xlabel("")
    #     ax1.set_ylabel("Min Goodput (Gb/s)")
    #     ax1.tick_params(axis="y")
    #     ax1.legend().remove()
        
    # ax2 = ax1.twinx()
    # if obj=="energy" or obj=="both":   
    #     # Secondary y-axis for GbJ
    #     sns.pointplot(
    #         data=combined_data, x="num_byte", y="GbJ", hue="nthreads", dodge=True,
    #         markers=[marker_map[a] for a in unique_nthreads],  # Map markers
    #         palette=["tab:green", "tab:green","tab:green","tab:green", "tab:green"], linestyle="--", ax=ax2, legend=False
    #     )
    #     ax2.set_ylabel("Gb/J")
    #     ax2.tick_params(axis="y")
    #     ax2.legend().remove()
        

    # ################ Create legend map ####################
    # legend_map = {}
    # keywords=["64", "128", "256", "512"]
    # for keyword in keywords:
    #     legend_map[keyword] =int(keyword)
    # ######################## END legend map ##################
    
    # # Custom legend
    # from matplotlib.lines import Line2D
    # legend_elements = [
    #     Line2D([0], [0], marker=marker_map[legend_map['64']], color='w', markerfacecolor="tab:blue", markersize=8, label='NTHREAD=64 [Gb/s]'),
    #     Line2D([0], [0],marker=marker_map[legend_map['64']], color='w', markerfacecolor='tab:green', markersize=8, label='NTHREAD=64 [Gb/J]'),
    #     Line2D([0], [0],marker=marker_map[legend_map['128']], color='w', markerfacecolor='tab:blue', markersize=8, label='NTHREAD=128 [Gb/s]'),
    #     Line2D([0], [0], marker=marker_map[legend_map['128']], color='w', markerfacecolor='tab:green', markersize=8, label='NTHREAD=128 [Gb/J]'),
    #     Line2D([0], [0], marker=marker_map[legend_map['256']], color='w', markerfacecolor='tab:blue', markersize=8, label='NTHREAD=256 [Gb/s]'),
    #     Line2D([0], [0], marker=marker_map[legend_map['256']], color='w', markerfacecolor='tab:green', markersize=8, label='NTHREAD=256 [Gb/J]'),
    #     Line2D([0], [0], marker=marker_map[legend_map['512']], color='w', markerfacecolor='tab:blue', markersize=8, label='NTHREAD=512 [Gb/s]'),
    #     Line2D([0], [0], marker=marker_map[legend_map['512']], color='w', markerfacecolor='tab:green', markersize=8, label='NTHREAD=512 [Gb/J]')
    # ]
    # ax1.legend(handles=legend_elements, title="")
    
    # # plt.title("Min Goodput vs. GbJ Across Different Byte Sizes")
    
    # # Create zoomed-in inset plot for the first 4 points
    # # Create zoomed-in inset plot for specific num_byte sizes
    # zoom_data = combined_data[combined_data["num_byte"].isin(["1B", "8B", "64B", "512B", "4KiB", "32KiB", "256KiB"])]
    # zoom_data['time_us']=zoom_data["time_ms"]*1000
    # zoom_data['device_energy [J]']=zoom_data["device_energy [MJ]"]*1e6 # energy in joule
    
    # axins = inset_axes(ax1, width="45%", height="55%", loc="center left", bbox_to_anchor=(0.09, -0.05, 1.2, 0.8), bbox_transform=ax1.transAxes)  # Move to left center
    # if obj=="perf" or  obj=="both":
    #     sns.pointplot(
    #         data=zoom_data, x="num_byte", y="time_ms", hue="nthreads", dodge=True, 
    #         markers=[marker_map[a] for a in unique_nthreads],  # Map markers
    #         palette=["tab:blue", "tab:blue", "tab:blue", "tab:blue", "tab:blue"], ax=axins, legend=False
    #     )

    # axins2 = axins.twinx()
    # if obj=="energy" or  obj=="both":
    #     sns.pointplot(
    #         data=zoom_data, x="num_byte", y="device_energy [J]", hue="nthreads", dodge=True,
    #         markers=[marker_map[a] for a in unique_nthreads],  # Map markers
    #         palette=["tab:green", "tab:green", "tab:green", "tab:green", "tab:green"], linestyle="--", ax=axins2, legend=False
    #     )

    # axins.set_xlabel("")
    # axins.set_ylabel("Time (ms)", fontsize=10)
    # if obj=="perf" or obj=="both":   
    #     axins.legend_.remove()

    # axins2.set_ylabel("Energy (J)", fontsize=10)
    # axins.tick_params(axis="both", which="both", labelsize=8)
    # axins2.tick_params(axis="both", which="both", labelsize=8)

    # # Adjust zoomed-in x limits based on data
    # axins.set_xlim(-0.1, 6.5)  # Ensuring the first 4 num_byte sizes are in view

    # # Indicate zoom area on main plot
    # mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="black", lw=1)
    # if obj=="energy" or obj=="both":   
    #     axins2.legend_.remove()


def main():
    parser = argparse.ArgumentParser(description="Generate plot comparing Baseline vs GPU-Aware approaches")
    parser.add_argument('--csv-dir', type=str, required=True, help="Directory containing the CSV files")
    parser.add_argument('--app', type=str, required=True, help="Select for which application generate the plot (e.g ar, a2a)")
    parser.add_argument('--obj', type=str, required=True, help="Select the objectives to plot (e.g perf, energy, both)")
    
    args = parser.parse_args()
    generate_plot(args.csv_dir, args.app, args.obj)

if __name__ == "__main__":
    main()


