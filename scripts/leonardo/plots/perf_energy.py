import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os

byte_mapping = {
    1: "1B",
    8: "8B",
    64: "64B",
    512: "512B",
    4 * 1024: "4KiB",
    32 * 1024: "32KiB",
    256 * 1024: "256KiB",
    16 * 1024 * 1024: "16MiB",
    128 * 1024 * 1024: "128MiB",
    1024 * 1024 * 1024: "1GiB"
}

def generate_plot(csv_dir):
    # File paths
    baseline_file = os.path.join(csv_dir, 'ar_baseline.csv')
    gpu_aware_file = os.path.join(csv_dir, 'ar_gpu_aware.csv')
    device_energy_file = os.path.join(csv_dir, 'ar_device_energy.csv')
    
    # Load data from CSV files
    baseline_data = pd.read_csv(baseline_file)
    gpu_aware_data = pd.read_csv(gpu_aware_file)
    gpu_energy_data = pd.read_csv(device_energy_file)
    combined_data = pd.concat([baseline_data, gpu_aware_data], ignore_index=True)
    combined_data = combined_data[combined_data['run'] == "run_avg"]
    combined_data = combined_data[['approach', 'chain_size', 'byte', 'host_energy_uj', 'min_goodput_Gbs']]
    combined_data['host_energy_uj'] = abs(combined_data['host_energy_uj']) / 1e+6
    combined_data.rename(columns={"host_energy_uj": "host_energy [J]"}, inplace=True)

    gpu_energy_data["energy [GJ]"] *= 1_000_000_000  # Convert GJ to J
    
    # Merge GPU energy data with combined_data
    combined_data = pd.merge(combined_data, gpu_energy_data[["approach", "byte", "energy [GJ]"]], 
                             on=["approach", "byte"], how="right") 
    combined_data.rename(columns={"energy [GJ]": "device_energy [J]"}, inplace=True)
    combined_data['device_energy [J]'] /= combined_data['chain_size']
    combined_data['GbMJ'] = (combined_data['byte'] / 1.25e+8) / ((combined_data['device_energy [J]'] + combined_data['host_energy [J]']) / 1e+6)
    combined_data['byte'] = combined_data['byte'].map(byte_mapping)
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.pointplot(
        data=combined_data, x="byte", y="min_goodput_Gbs", hue="approach", dodge=True, markers=["s", "o"], 
        palette=["tab:blue", "tab:blue"], ax=ax1
    )
    ax1.set_xlabel("")
    ax1.set_ylabel("Min Goodput (Gb/s)")
    ax1.tick_params(axis="y")
    
    # Secondary y-axis for GbMJ
    ax2 = ax1.twinx()
    sns.pointplot(
        data=combined_data, x="byte", y="GbMJ", hue="approach", dodge=True, markers=["s", "o"], 
        palette=["tab:green", "tab:green"], linestyle="--", ax=ax2, legend=False
    )
    ax2.set_ylabel("Gb/MJ")
    ax2.tick_params(axis="y")
    ax2.legend().remove()  # Removes the legend
    ax1.legend().remove()  # Removes the legend

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue', markersize=8, label='Baseline [Gb/s]'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:green', markersize=8, label='Baseline [Gb/MJ]'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='tab:blue', markersize=8, label='GPU-Aware [Gb/s]'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='tab:green', markersize=8, label='GPU-Aware [Gb/MJ]')
    ]
    ax1.legend(handles=legend_elements, title="")
    
    plt.title("Min Goodput vs. GbMJ Across Different Byte Sizes")
    plt.savefig("prova.pdf")
    plt.show()
    

def main():
    parser = argparse.ArgumentParser(description="Generate plot comparing Baseline vs GPU-Aware approaches")
    parser.add_argument('--csv-dir', type=str, required=True, help="Directory containing the CSV files")
    args = parser.parse_args()
    generate_plot(args.csv_dir)

if __name__ == "__main__":
    main()
