import argparse
import re
import os
from pathlib import Path

# logs/perf/library/collective/tuning_param/slurm_file
tuning_parameters_map={
    "nchannels": "NCCL_MIN/MAX_CTAS",
    "nthreads": "NCCL_NTHREADS",
    "launch": "NCCL_LAUNCH_MODE",
    "buffsize":"NCCL_BUFFSIZE"
}
# collectives=["ar","a2a"]
collectives=["ar"]# TODO: Add all collectives

# tuning_parameters=["nchannels", "nthreads", "launch", "buffsize"] 
tuning_parameters=["launch"]
libraries=["nccl"] # TODO: Add other libraries

def parse_file(tuning_dir, out_dir, lib, coll, tuning_param):
    os.makedirs(out_dir, exist_ok=True)

    tuning_value = None
    approach_printed = False
    output_filename=Path(os.path.join(out_dir, f'{lib}_{coll}_{tuning_param}.csv'))
    with open(output_filename, 'w') as outfile:
        for file_path in tuning_dir.glob("*.out"):
            print(file_path)
            with open(file_path, 'r') as infile:
                for line in infile:
                    split_line = line.strip().split(',')
                    # Check if the line contains 'Running with NCCL_* '
                    # match = re.search(r'Running with (\d+) threads per block', line)
                    pattern = rf'Running with {re.escape(tuning_parameters_map[tuning_param])}\s+(\d+)'

                    if "launch" in tuning_param:
                        pattern = rf'Running with {re.escape(tuning_parameters_map[tuning_param])}\s+(\w+)'
                                                
                    # Dynamically build the regex pattern using the value of 'tuning'
                    match = re.search(pattern, line)
                    if match:
                        tuning_value = match.group(1)

                    # Check if the split line has more than 1 element and if the first element is not 'approach'
                    if len(split_line) > 1:
                        if 'approach' not in split_line[0].lower():
                            if tuning_value is not None:
                                outfile.write(f"{line.strip()},{tuning_value}\n")
                        else:
                            # If 'approach' is found in the first split part, only write it once
                            if not approach_printed:
                                if tuning_value is not None:
                                    outfile.write(f"{line.strip()},{tuning_param}\n")
                                approach_printed = True


def main():
    parser = argparse.ArgumentParser(description="Parse an input file and extract thread data.")
    parser.add_argument("--in-dir", required=True, help="Path to the dircotry containing the slurm output for all the benchmakrs")
    parser.add_argument("--out-dir", default="output.csv", help="Path to the direcotry with the final csv file for each library, collective and tuning parameter")

    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    
   
    for lib in libraries:
        library_path=Path(os.path.join(in_dir, lib))
        out_dir_lib= Path(os.path.join(out_dir, lib))
        for coll in collectives:
            coll_path=Path(os.path.join(library_path, coll))
            out_dir_coll=Path(os.path.join(out_dir_lib, coll))
            
            for tuning_param in tuning_parameters:
                tuning_path=Path(os.path.join(coll_path, tuning_param))
                out_dir_tuning=Path(os.path.join(out_dir_coll, tuning_param))
                parse_file(tuning_path,out_dir_tuning, lib, coll, tuning_param)

    
    # parse_file(args.in_dir, args.out_dir)

if __name__ == "__main__":
    main()
