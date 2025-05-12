import argparse
import re

def parse_file(input_filename, output_filename):
    nthreads = None
    approach_printed = False

    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        for line in infile:
            split_line = line.strip().split(',')

            # Check if the line contains 'Running with X threads per block'
            match = re.search(r'Running with (\d+) threads per block', line)
            if match:
                nthreads = match.group(1)

            # Check if the split line has more than 1 element and if the first element is not 'approach'
            if len(split_line) > 1:
                if 'approach' not in split_line[0].lower():
                    if nthreads is not None:
                        outfile.write(f"{line.strip()},{nthreads}\n")
                else:
                    # If 'approach' is found in the first split part, only write it once
                    if not approach_printed:
                        if nthreads is not None:
                            outfile.write(f"{line.strip()},nthreads\n")
                        approach_printed = True

def main():
    parser = argparse.ArgumentParser(description="Parse an input file and extract thread data.")
    parser.add_argument("--input-file", required=True, help="Path to the input file")
    parser.add_argument("--output-file", default="output.csv", help="Path to the output file (default: output.txt)")

    args = parser.parse_args()

    parse_file(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
