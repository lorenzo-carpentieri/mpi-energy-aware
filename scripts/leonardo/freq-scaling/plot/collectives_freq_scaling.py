import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
import argparse
import os
import io 
def process_csv(file_path):
    """Process a single CSV file: remove unnecessary lines, extract NCHANNELS, and add it as a column."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Find the first line that starts with "Running"
    start_index = next((i for i, line in enumerate(lines) if line.startswith("Running")), None)

    if start_index is None:
        raise ValueError(f"No 'Running' line found in {file_path}")

    # Extract NCHANNELS value (it's in the "Running" line)
    nchannels = int(lines[start_index].strip().split()[-1])  # Last number in the line

    # The actual CSV starts two lines after "Running" (skip header too)
    csv_data = "\n".join(lines[start_index + 1:])

    # Read the CSV data
    df = pd.read_csv(io.StringIO(csv_data))

    # Add the NCHANNELS column
    df["nchannels"] = nchannels

    return df
def process_folder(folder_path):
    """Process all CSV files in a folder and concatenate them into one DataFrame."""
    all_dfs = pd.DataFrame()
    for filename in os.listdir(folder_path):
        if filename.endswith(".out"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing {file_path} ...")
            try:
                df = process_csv(file_path)
                all_dfs = pd.concat([all_dfs, df], axis=0, ignore_index=True)

            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")
    print(all_dfs)
    return all_dfs


def freq_scaling_plot(df):
    """
    Plots a scatter plot with speedup on the x-axis, normalized energy on the y-axis,
    color-coded by core frequency, and different marker types based on nchannels.
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing the following columns:
            - 'speedup': The speedup values.
            - 'energy': The energy consumption values.
            - 'baseline': The baseline energy values.
            - 'core_frequency': The frequency of the core.
            - 'nchannels': The number of channels, used for marker style.
    
    """
    
    df = df[df['core_freq'] == "run_avg"]
    baseline_row = df[df['run'] == 1260] # change with default freq.
   
    # Create a dictionary mapping nchannels to baseline values for the baseline core-freq
    baseline_dict = baseline_row.set_index('nchannels')['energy_j'].to_dict()
    print(baseline_dict)
    # Define a function to normalize energy for each row based on its nchannels
    def normalize(row):
        baseline = baseline_dict.get(row['nchannels'], 1)  # Default to 1 if baseline not found
        return row['energy_j'] / baseline

    # Apply normalization across the DataFrame
    df['normalized_energy'] = df.apply(normalize, axis=1)
    exit()
    # Define marker styles based on 'nchannels' values (you can customize this)
    markers = {1: 'o', 2: 's', 3: '^', 4: 'D', 5: 'X'}
    
    # Set up the plot
    plt.figure(figsize=(10, 6))
    
    # Create a scatterplot
    scatter = sns.scatterplot(
        data=df,
        x='speedup',
        y='normalized_energy',
        hue='core_freq',  # Color by core frequency
        style='nchannels',  # Different markers for each nchannels value
        palette='viridis',  # You can choose different palettes
        markers=markers,  # Custom marker styles
        s=100,  # Marker size
        legend='full'
    )
    
    # Add a color bar
    norm = mpl.colors.Normalize(vmin=df['core_freq'].min(), vmax=df['core_freq'].max())
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])  # Empty array, needed for the colorbar to work
    cbar = plt.colorbar(sm, ax=scatter.axes)
    cbar.set_label('Core Frequency')  # Label for the color bar

    # Set plot labels
    plt.xlabel('Speedup')
    plt.ylabel('Normalized Energy (Energy / Baseline)')
    plt.title('Speedup vs Normalized Energy')

    # Show the color bar for core frequency
    plt.legend(title='Core Frequency', loc='upper left')
    plt.show()
    
    
def main():
    parser = argparse.ArgumentParser(description="Process and merge CSV files from a folder.")
    parser.add_argument("--slurm-logs-folder", type=str, help="Path to the folder containing csv file")
    
    args = parser.parse_args()
    
    all_data_df = process_folder(args.slurm_logs_folder)

    freq_scaling_plot(all_data_df)
    
if __name__ == "__main__":
    main()
