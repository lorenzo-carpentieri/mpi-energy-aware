import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

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

def generate_plot(csv_dir, app):
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
    combined_data['device_energy [MJ]'] *= num_gpus
    combined_data['GbJ'] = (combined_data['num_byte'] / 1.25e+8) / (combined_data['device_energy [MJ]']*1e6) # TODO: consider host time/energy
    
    combined_data['device_energy [MJ]'] /= combined_data['chain_size']
    combined_data['device_energy [MJ]'] *= num_gpus
    combined_data['GbJ'] = (combined_data['num_byte'] / 1.25e+8) / (combined_data['device_energy [MJ]']*1e6) # TODO: consider host time/energy
    combined_data['num_byte'] = combined_data['num_byte'].map(byte_mapping)
    combined_data["edp"] =  (combined_data['time_ms']/1000)*(combined_data['device_energy [MJ]']*1e6) # second * joule

    combined_data = combined_data[combined_data["run"]=='run_avg']
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
    
    # Create main plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.pointplot(
        data=combined_data, x="num_byte", y="edp", hue="nthreads", dodge=True, 
        markers=[marker_map[a] for a in unique_nthreads],  # Map markers
        palette=[palette_map[a] for a in unique_nthreads], # Map colors
        ax=ax1
    )

    ax1.set_xlabel("")
    ax1.set_ylabel("EDP (s*J)")
    ax1.tick_params(axis="y")
    ax1.legend().remove()
     

    ################ Create legend map ####################
    legend_map = {}
    keywords=["64", "128", "256", "512"]
    for keyword in keywords:
        legend_map[keyword] =int(keyword)
    ######################## END legend map ##################
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker=marker_map[legend_map['64']], color='w', markerfacecolor=palette_map[legend_map['64']], markersize=8, label='NTHREAD=64 [Gb/s]'),
        Line2D([0], [0],marker=marker_map[legend_map['128']], color='w', markerfacecolor=palette_map[legend_map['128']], markersize=8, label='NTHREAD=128 [Gb/s]'),
        Line2D([0], [0], marker=marker_map[legend_map['256']], color='w', markerfacecolor=palette_map[legend_map['256']], markersize=8, label='NTHREAD=256 [Gb/s]'),
        Line2D([0], [0], marker=marker_map[legend_map['512']], color='w', markerfacecolor=palette_map[legend_map['512']], markersize=8, label='NTHREAD=512 [Gb/s]'),
    ]
    ax1.legend(handles=legend_elements, title="")
    
    # plt.title("Min Goodput vs. GbJ Across Different Byte Sizes")
    
    # Create zoomed-in inset plot for the first 4 points
    # Create zoomed-in inset plot for specific num_byte sizes
    zoom_data = combined_data[combined_data["num_byte"].isin(["1B", "8B", "64B", "512B", "4KiB", "32KiB", "256KiB"])]
    zoom_data['time_us']=zoom_data["time_ms"]*1000
    zoom_data['device_energy [J]']=zoom_data["device_energy [MJ]"]*1e6 # energy in joule
    zoom_data['edp']=zoom_data['time_us']*zoom_data['device_energy [MJ]']*1e6 # edp as ms * J
    
    axins = inset_axes(ax1, width="45%", height="55%", loc="center left", bbox_to_anchor=(0.09, -0.05, 1.2, 0.8), bbox_transform=ax1.transAxes)  # Move to left center
    sns.pointplot(
        data=zoom_data, x="num_byte", y="edp", hue="nthreads", dodge=True, 
        markers=[marker_map[a] for a in unique_nthreads],  # Map markers
        palette=[palette_map[a] for a in unique_nthreads],  
        ax=axins, legend=False
    )

   

    axins.set_xlabel("")
    axins.set_ylabel("Time (ms)", fontsize=10)
    axins.legend_.remove()

    axins.tick_params(axis="both", which="both", labelsize=8)

    # Adjust zoomed-in x limits based on data
    axins.set_xlim(-0.1, 6.5)  # Ensuring the first 4 num_byte sizes are in view

    # Indicate zoom area on main plot
    mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="black", lw=1)

    plt.savefig(f"{app}.pdf")

def main():
    parser = argparse.ArgumentParser(description="Generate plot comparing Baseline vs GPU-Aware approaches")
    parser.add_argument('--csv-dir', type=str, required=True, help="Directory containing the CSV files")
    parser.add_argument('--app', type=str, required=True, help="Select for which application generate the plot (e.g ar, a2a)")
    
    args = parser.parse_args()
    generate_plot(args.csv_dir, args.app)

if __name__ == "__main__":
    main()
