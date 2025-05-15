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


def print_pareto(df, x_obj, y_obj, pareto_line_color, pareto_label_name):
    # Compute the pareto set point and print on the plot
    # Creaete a data frame with energy and speedup
    df_xobj_yobj = pd.DataFrame({"x_obj": df[x_obj], "y_obj": df[y_obj]})
    mask = paretoset(df_xobj_yobj, sense=["max", "min"])
    pset = df_xobj_yobj[mask]
    print(pset)
    pset = pset.sort_values(by=["x_obj"])
    
    np_array = pset.to_numpy()
    pset_size = len(pset["x_obj"])
    
    cur_xlim_left, cur_xlim_right = plt.xlim()
    cur_xlim_bottom, cur_ylim_top = plt.ylim()
    x1, y1 = [cur_xlim_left, np_array[0][0]], [np_array[0][1], np_array[0][1]]
    plt.plot(x1, y1, color=pareto_line_color, linewidth=2.5, label="Pareto-front")

    for i in range(pset_size):
        if not (i == pset_size-1):
            current_x = np_array[i][0]
            current_y = np_array[i][1]
            next_x = np_array[i+1][0]
            next_y = np_array[i+1][1]
            x1, y1 = [current_x, current_x], [current_y, next_y]
            x2, y2 = [current_x, next_x], [next_y, next_y]
            plt.plot(x1, y1, x2, y2, color=pareto_line_color, linewidth=2.5)



def generate_plot(df, out_dir, app):
    os.makedirs(out_dir, exist_ok=True)
        
    
    
   
    # Ensure both columns have the same type
    df["num_byte"] = df["num_byte"].astype(int)

    df["approach"] = df["approach"].astype(str)
    
    # device energy in MJ
    df['device_energy [MJ]']= df['device_energy']/ 1_000_000
    df['GbJ'] = (df['num_byte'] / 1.25e+8) / (df['device_energy [MJ]']*1e6)
    
    # TODO: add host energy
   
    df = df[df["run"]=='run_avg']
    df = df[df["approach"].str.contains(app+"_", na=False)] # Extract the data related to the collective specified by app
    # Create a new column for hue and style
    df["Protocols x Algorithms"] = df["prot"] + " - " + df["alg"]
    df["Threads x Channels"] = df["threads"].astype(str) + " - " + df["channels"].astype(str)
    df["Host Energy (J)"] = df["host_energy"] / 1000 # millijoule to joule
    df.to_csv(f"{out_dir}/nccl_{app}_all_params.csv", index=False)

    # type can be host or device
    def host_device_energy_plot(df, type):
        
        # Unique message sizes
        msg_sizes = sorted(df["num_byte"].unique())
        n_sizes = len(msg_sizes)
        # Grid size: 3 columns, ceil(n_sizes / 3) rows
        ncols = 3
        nrows = math.ceil(n_sizes / ncols)
        # Create subplots
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)
        

        
        for i, msg_size in enumerate(msg_sizes):
            row = i // ncols
            col = i % ncols
            ax = axes[row][col]
            filtered_data = df[df["num_byte"]==msg_sizes[i]]
            x_obj="Min Goodput (Gb/s)"
            y_obj="Device Energy (J)"
            filtered_data.loc[:, "Device Energy (J)"] = filtered_data['device_energy [MJ]'] * 1_000_000  

            if "device" == type:
                y_obj="Device Energy (J)"
            elif "host" == type: 
                y_obj="Host Energy (J)"
                # filtered_data = filtered_data[filtered_data[y_obj] >= 0]
            else:
                y_obj="Host and Device Energy (J)"
                filtered_data[y_obj]=filtered_data["Host Energy (J)"] + filtered_data["Device Energy (J)"]
            
            filtered_data.loc[:, x_obj] = filtered_data['min_goodput_Gbs']
            
            # For readibility we use different unit measure for each plot
            if i < 6:
                x_obj="Min Goodput Mb/s"
                filtered_data[x_obj] = (filtered_data['min_goodput_Gbs'] * 1_000)
            
            filtered_data[x_obj] = filtered_data[x_obj].round(3)

            sns.scatterplot(
                data=filtered_data,
                x=x_obj,
                y=y_obj,
                hue="Protocols x Algorithms",
                style="Threads x Channels",
                palette="tab10",
                ax=ax,
                s=100,
                edgecolor="black"
            )

            print_pareto(filtered_data, x_obj, y_obj, "red", "Pareto-front")
            ax.set_title(f"Size {byte_mapping[int(msg_size)]}")
            ax.set_xlabel(x_obj)
            ax.set_ylabel(y_obj)
            ax.legend_.remove()
            ax.grid(True)

        # Remove unused subplots if any
        for j in range(i + 1, nrows * ncols):
            fig.delaxes(axes[j // ncols][j % ncols])

        # Move legend outside of plot grid
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.subplots_adjust(right=0.85)  # Leave space for legend
        plt.savefig(f"{out_dir}/nccl_{app}_all_params_{type}_energy.pdf")
        plt.clf()      # Clears the current figure
        plt.cla()      # Clears the current axes
        plt.close()   

    host_device_energy_plot(df, "host")
    host_device_energy_plot(df, "device")
    host_device_energy_plot(df, "host_device")
    
    
    # ################### Generate marker and palette for each approach ###################################
    # # Get unique approaches 
    # unique_tuning_params = combined_data[tuning_param].unique()

    # # Define available colors and markers
    # available_colors = sns.color_palette("tab20", len(unique_tuning_params))  # Get a list of distinct colors
    # available_markers = list(itertools.islice(itertools.cycle(all_markers), len(unique_tuning_params)))

    # # Create mapping dictionaries
    # palette_map = {nthread: color for nthread, color in zip(unique_tuning_params, available_colors)}
    # marker_map = {nthread: marker for nthread, marker in zip(unique_tuning_params, available_markers)}
    # ######################################### END marker and palette generation ###########################
    # print(marker_map)
    
    # filter_num_byte=[8, 512, 32768, 262144, 2097152, 134217728, 1073741824]
    
    # figureSize=(25,3.5)
    # if "buffsize" in tuning_param:
    #     figureSize=(30,4.5)
   
        
    # fig, axes = plt.subplots(1, len(filter_num_byte), figsize=figureSize)  # Adjust figsize as needed
    
    # fig.suptitle(f"{lib}/{coll}/{tuning_param}", fontsize=14, fontweight='bold')

    # # filter_num_byte=['8B', '512B', '32KiB', '1GiB']
    
    # for i, ax in enumerate(axes):
    #     # take only 8B, 512B, 32KiB, 16MiB, 1GiB
    #     filtered_data = combined_data[combined_data["num_byte"]==filter_num_byte[i]]
    #     x_obj="Min Goodput (Gb/s)"
    #     y_obj="Energy (J)"
    #     filtered_data.loc[:, y_obj] = filtered_data['device_energy [MJ]'] * 1_000_000  
    #     filtered_data.loc[:, x_obj] = filtered_data['min_goodput_Gbs']
      
    #     # For readibility we use different unit measure for each plot
    #     if i < 3:
    #         x_obj="Min Goodput Mb/s"
    #         print(x_obj)
    #         filtered_data[x_obj] = (filtered_data['min_goodput_Gbs'] * 1_000).round(2)

    #     # filtered_data[tuning_param] = filtered_data[tuning_param].astype(int)
    #     filtered_data = filtered_data[[tuning_param, x_obj, y_obj]]
    #     filtered_data = filtered_data.reset_index(drop=True)
    #     if "launch" in tuning_param:
    #         filtered_data[tuning_param] = filtered_data[tuning_param].astype(str)
    #     else:
    #         filtered_data[tuning_param] = filtered_data[tuning_param].astype(int)

    #     print(filtered_data)

    #     sns.scatterplot(
    #         data=filtered_data, x=x_obj, y=y_obj, 
    #         hue=tuning_param, 
    #         palette='tab20',
    #         ax=ax
    #     )
    #     ax.set_title(f"Message size: {byte_mapping[filter_num_byte[i]]}")
    #     # ax.legend(ncol=2, title=tuning_param)
    #     # Get current handles and labels from the plot
    #     if "buffsize" in tuning_param:       
    #         handles, labels = ax.get_legend_handles_labels()
    #         label_map = {}

    #         for val in unique_tuning_params:
    #             num = int(val)  # Make sure it's a standard Python int
    #             if num <= 4096:
    #                 label = f"{num}B"
    #             elif num <= 524288:
    #                 label = f"{num // 1024}KiB"
    #             else:
    #                 label = f"{round(num / (1024 * 1024))}MiB"
    #             label_map[int(num)] = label 

            
    #         # Apply the new labels using list comprehension
    #         new_labels = [label_map.get(int(label)) for label in labels]
            
    #         # Recreate the legend with modified labels
    #         ax.legend(handles, new_labels, ncol=2, title=tuning_param)

    # plt.tight_layout()
    # plt.savefig(f"{out_dir}/{coll}_{lib}_{tuning_param}.pdf")
    

def main():
    parser = argparse.ArgumentParser(description="NCCL energy characterization with different parameters")
    parser.add_argument('--log-dir', type=str, required=True, help="Directory containing the PERFORMAANCE data related to each library collective and tuning parameter. e.g nccl, ar, nthreads")
    parser.add_argument('--out-dir', type=str, required=True, help="path to the plot directory. The script generate in this directory new path. e.g. out-dir/lib/collectives/tuning_parameter/ar_lib_tuning_parameter.pdf")
    parser.add_argument('--app', type=str, required=True, help="Select for which application generate the plot (e.g ar, a2a)")
    
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir)
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
    generate_plot(final_df, out_dir, args.app)
   
 
if __name__ == "__main__":
    main()


