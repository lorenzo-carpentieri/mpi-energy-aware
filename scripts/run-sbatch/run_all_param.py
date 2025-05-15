import argparse
import os
import subprocess
from pathlib import Path
import shutil 

tuning_parameters_map={
    "nchannels": "NCCL_MIN/MAX_CTAS",
    "nthreads": "NCCL_NTHREADS",
    "launch": "NCCL_LAUNCH_MODE",
    "buffsize":"NCCL_BUFFSIZE"
}


#TODO: Change this dictionary depending on the NCCL version. Some algortihms are only available for higher version of nccl.
algorithms={
    "ar": ["ring", "collnetdirect", "tree"],
    # "a2a": ["ring", "collnetdirect", "tree","collnet","collnetchain","nvls","nvlstree","pat"],
} 


# Protocols that can be used for each algorithm and collective
protocols={
    "ring": {"ar": ["LL", "LL128", "Simple"], "a2a": ["LL", "LL128", "Simple"]},
    "tree": {"ar": ["LL", "LL128", "Simple"],"a2a": ["LL", "LL128", "Simple"]},
    "collnetdirect":  {"ar": ["LL", "LL128", "Simple"],"a2a": ["LL", "LL128", "Simple"]},
    # "collnet": {"ar": ["LL", "LL128", "Simple"],"a2a": ["LL", "LL128", "Simple"]},
    # "collnetchain": {"ar": ["LL", "LL128", "Simple"],"a2a": ["LL", "LL128", "Simple"]},
    # "nvls": {"ar": ["LL", "LL128", "Simple"],"a2a": ["LL", "LL128", "Simple"]},
    # "nvlstree": {"ar": ["LL", "LL128", "Simple"],"a2a": ["LL", "LL128", "Simple"]},
    # "pat": {"ar": ["LL", "LL128", "Simple"],"a2a": ["LL", "LL128", "Simple"]},
} # 3

nthreads= ["64","128", "256", "512"] # 4 
nchannels= ["2", "4", "8", "16", "32"] # 5




# collectives=["ar","a2a"]
collectives=["ar"]# TODO: Add all collectives

# tuning_parameters=["nchannels", "nthreads", "launch", "buffsize"] 
tuning_parameters=["launch"]
libraries=["nccl"] # TODO: Add other libraries


def backup_log_dirs(base_log_dir):
    for category in ['perf', 'power']:
        source = Path(base_log_dir) / category
        backup = Path(base_log_dir) / f"{category}-bp"
        
        if source.exists():
            if backup.exists():
                shutil.rmtree(backup)  # Remove old backup if exists
            shutil.copytree(source, backup)
            print(f"Copied {source} to {backup}")
        else:
            print(f"Source directory {source} does not exist, skipping backup.")

# Define tuning-specific parameters
def generate_tuning_parameters():
    # Generate buffer sizes from 2^10 (1 KiB) to 2^23 (8 MiB)
    buff_sizes = [[str(2 ** exp)] for exp in range(10, 24)]  # 10 to 23 inclusive

    tuning_parameters = {
        "nchannels": [["2"], ["4"], ["8"], ["16"], ["32"]],
        "launch": [["PARALLEL"], ["GROUP"]],
        "buffsize": buff_sizes,
        "nthreads": [["64"], ["128"], ["256"], ["512"]],
        "default": [[]]
    }

    return tuning_parameters

def parse_args():
    parser = argparse.ArgumentParser(description="Process SBATCH scripts and manage logs.")
    parser.add_argument('--sbatch-script-file', required=True, help='SBATCH script file ')
    parser.add_argument('--log-dir', required=True, help='Base directory for log output')

    return parser.parse_args()

def create_log_dirs(base_log_dir, library, collective):
    for category in ['perf', 'power']:
        path = Path(base_log_dir) / category / library / collective
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created log directory: {path}")

def parse_script_name(script_name):
    """
    Expected format: library_collective_tuningtype.sh
    Example: nccl_allreduce_blocksize.sh
    """
    base_name = script_name.stem
    parts = base_name.split('_')
    if len(parts) != 3:
        raise ValueError(f"Invalid script name format: {script_name}")
    return parts  # returns (library, collective, tuning_type)

def main():
    args = parse_args()
    script_file = Path(args.sbatch_script_file)
    log_dir = Path(args.log_dir)
    fixed_var = args.fix_var
    
    
    backup_log_dirs(log_dir)
    
    # For each NCCL varaible generate different values
    tuning_parameters = generate_tuning_parameters()

    for lib in libraries:
        for coll in collectives:
            # Fix algorithm
            for alg in algorithms[coll]:
                # Fix protocol 
                for prot in protocols[alg][coll]:
                    # Fix nthreads
                    for thread in nthreads:
                        for channel in nchannels:
                            create_log_dirs(log_dir, lib, coll)
                            print(f"Executing script: {script_file}")
                            print(f"Submitting {script_file.name} with args: {alg}, {prot}, {thread}, {channel}")
                            subprocess.run(["sbatch", str(script_file), alg, prot, thread, channel], check=True)

                            
                            
    
if __name__ == "__main__":
    main()
