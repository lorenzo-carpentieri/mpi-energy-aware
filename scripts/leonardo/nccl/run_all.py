import argparse
import os
import subprocess
from pathlib import Path
import shutil 

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
    parser.add_argument('--sbatch-script-dir', required=True, help='Directory containing .sh SBATCH scripts')
    parser.add_argument('--log-dir', required=True, help='Base directory for log output')
    return parser.parse_args()

def create_log_dirs(base_log_dir, library, collective, tuning_type):
    for category in ['perf', 'power']:
        path = Path(base_log_dir) / category / library / collective / tuning_type
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
    script_dir = Path(args.sbatch_script_dir)
    log_dir = Path(args.log_dir)
    
    backup_log_dirs(log_dir)
    
    # For each NCCL varaible generate different values
    tuning_parameters = generate_tuning_parameters()

    for script_path in script_dir.glob("*.sh"):
        try:
            library, collective, tuning_type = parse_script_name(script_path)
            create_log_dirs(log_dir, library, collective, tuning_type)
            if "launch" not in tuning_type:
                continue
            
           
                
            print(f"Executing script: {script_path}")
             # Get argument sets for this tuning_type
            argument_sets = tuning_parameters.get(tuning_type, tuning_parameters["default"])
            
            for args in argument_sets:
                print(f"Submitting {script_path.name} with args: {' '.join(args)}")
                subprocess.run(["sbatch", str(script_path), *args], check=True)

        except Exception as e:
            print(f"Error processing {script_path.name}: {e}")

if __name__ == "__main__":
    main()
