import argparse
import subprocess
import re
import numpy as np

def read_frequencies(file_path):
    """Read frequencies from a file, assuming one frequency per line."""
    with open(file_path, 'r') as f:
        return [float(line.strip()) for line in f if line.strip()]

def sample_frequencies(frequencies, sampling_factor):
    """Sample frequencies based on the given factor."""
    return frequencies[::sampling_factor]

def run_sbatch(frequencies, n_channels, sbatch_script):
    """Run the SBATCH script with given frequencies and channels."""
    for freq in frequencies:
        with open(sbatch_script, 'r') as file:
            script_content = file.read()

        # Define the pattern to match '--gpu-freq=' followed by an integer
        pattern = r'(--gpu-freq=)(\d+(\.\d*)?|\.\d+)'

        # Replace the matched pattern with '--gpu-freq=' followed by core_freq
        modified_content = re.sub(pattern, r'\g<1>' + str(int(freq)), script_content)
        
        # Write the modified content back to the script file
        with open(sbatch_script, 'w') as file:
            file.write(modified_content)
            
        for i in n_channels:
            cmd = ["sbatch", sbatch_script, str(freq), str(i)]
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Run SBATCH script with sampled frequencies.")
    parser.add_argument("--freq-file", help="Path to the file containing frequencies.")
    parser.add_argument("--sampling-factor", type=int, help="Factor for frequency sampling.")
    parser.add_argument("--sbatch-script", help="Path to the SBATCH script.")
    parser.add_argument("--nchannels-factor", type=int, default=1, help="Number of channels (default: 32).")
    
    args = parser.parse_args()
    nchannels = np.arange(32)  # Creates an array with values from 0 to 31
    nchannels_factor=int(args.nchannels_factor)
    nchannels[0]+=1
    nchannels = nchannels[::nchannels_factor]
    frequencies = read_frequencies(args.freq_file)
    sampled_frequencies = sample_frequencies(frequencies, args.sampling_factor)
    run_sbatch(sampled_frequencies, nchannels, args.sbatch_script)

if __name__ == "__main__":
    main()
