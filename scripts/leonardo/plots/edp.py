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
    num_gpus=4 # we have a node with 4 gpus
    cuda_time_file = os.path.join(csv_dir, 'all_mpi_cuda_time.csv')
    cuda_power_file = os.path.join(csv_dir, 'all_mpi_cuda_power.csv')
    
    cuda_time_df= pd.read_csv(cuda_time_file)
    cuda_power_df= pd.read_csv(cuda_power_file)
    
    # Ensure both columns have the same type
    cuda_time_df["num_byte"] = cuda_time_df["num_byte"].astype(float)
    cuda_power_df["num_byte"] = cuda_power_df["num_byte"].astype(float)

    cuda_time_df["approach"] = cuda_time_df["approach"].astype(str)
    cuda_power_df["approach"] = cuda_power_df["approach"].astype(str)
    # Now merge
    combined_data = pd.merge(cuda_time_df, cuda_power_df[["approach", "num_byte", "energy [MJ]"]],
                            on=["approach", "num_byte"], how="right")
    
    combined_data["chain_size"] = combined_data["chain_size"].astype(float)
    combined_data.rename(columns={"energy [MJ]": "device_energy [MJ]"}, inplace=True)
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
    unique_approaches = combined_data["approach"].unique()

    # Define available colors and markers
    available_colors = sns.color_palette("tab10", len(unique_approaches))  # Get a list of distinct colors
    available_markers = ["o", "s", "^", "D", "v", "P", "*", "X"]  # Expand if needed

    # Create mapping dictionaries
    palette_map = {approach: color for approach, color in zip(unique_approaches, available_colors)}
    marker_map = {approach: marker for approach, marker in zip(unique_approaches, available_markers)}
  
    ######################################### END marker and palette generation ###########################
        
    # Create main plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.pointplot(
        data=combined_data, x="num_byte", y="edp", hue="approach", dodge=True,
        markers=[marker_map[a] for a in unique_approaches],  # Map markers
        palette=[palette_map[a] for a in unique_approaches], # Map colors
        ax=ax1
    )

    ax1.set_xlabel("")
    ax1.set_ylabel("EDP (s*J)")
    ax1.tick_params(axis="y")
    
    ################ Create legend map ####################
    legend_map = {}
    keywords=["Baseline", "GPU-Aware", "NCCL"]
    for keyword in keywords:
        # Find the first key in palette_map that contains the keyword (case-insensitive)
        if keyword =="GPU-Aware":
            matched_key = next((k for k in palette_map if "aware" in k.lower()), None)
        else:
            matched_key = next((k for k in palette_map if keyword.lower() in k.lower()), None)
        keyword.replace("-", " ")
        if matched_key:
            legend_map[keyword] = matched_key
    ######################## END legend map ##################
    # Custom legend
    print(legend_map)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker=marker_map[legend_map['Baseline']], color=palette_map[legend_map['Baseline']], markerfacecolor=palette_map[legend_map['Baseline']], markersize=8, label='Baseline'),
        Line2D([0], [0], marker=marker_map[legend_map['GPU-Aware']], color=palette_map[legend_map['GPU-Aware']], markerfacecolor=palette_map[legend_map['GPU-Aware']], markersize=8, label='GPU-Aware'),
        Line2D([0], [0], marker=marker_map[legend_map['NCCL']], color=palette_map[legend_map['NCCL']], markerfacecolor=palette_map[legend_map['NCCL']], markersize=8, label='NCCL'),
    ]
    ax1.legend(handles=legend_elements, title="")
    
    plt.title("EDP Across Different Byte Sizes")
    
    # Create zoomed-in inset plot for the first 4 points
    # Create zoomed-in inset plot for specific num_byte sizes
    zoom_data = combined_data[combined_data["num_byte"].isin(["1B", "8B", "64B", "512B", "4KiB", "32KiB", "256KiB"])]
    zoom_data['edp']=zoom_data['time_ms']*zoom_data['device_energy [MJ]']*1e6 # edp as ms * J
    
    print(zoom_data)
    axins = inset_axes(ax1, width="45%", height="45%", loc="center left", bbox_to_anchor=(0.11, -0.05, 1, 1), bbox_transform=ax1.transAxes)  # Move to left center

    sns.pointplot(
        data=zoom_data, x="num_byte", y="edp", hue="approach", dodge=True,
        markers=[marker_map[a] for a in unique_approaches],  # Map markers
        palette=[palette_map[a] for a in unique_approaches], # Map colors
        ax=axins, legend=False
    )
    


    axins.set_xlabel("")
    axins.set_ylabel("EDP (ms*J)", fontsize=10)
    
    axins.legend_.remove()
    # ax2 = ax1.twinx()

    # Adjust zoomed-in x limits based on data
    # axins.set_xlim(-0.1, 4.5)  # Ensuring the first 4 num_byte sizes are in view

    # Indicate zoom area on main plot
    # mark_inset(ax1, axins, loc1=1, loc2=1, fc="none", ec="black", lw=1)

    plt.savefig(f"{app}_edp.pdf")

def main():
    parser = argparse.ArgumentParser(description="Generate plot comparing Baseline vs GPU-Aware approaches")
    parser.add_argument('--csv-dir', type=str, required=True, help="Directory containing the CSV files")
    parser.add_argument('--app', type=str, required=True, help="Select for which application generate the plot (e.g ar, a2a)")
    args = parser.parse_args()
    generate_plot(args.csv_dir, args.app)

if __name__ == "__main__":
    main()
