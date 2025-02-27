NVCC = nvcc
CFLAGS = -lmpi -lnvidia-ml -lnccl -DNVML -I/leonardo/home/userexternal/lcarpent/mpi-energy/mpi-energy-aware/cuda/power-profiler
SRC_DIR = $(CURDIR)/cuda
EXE_DIR = $(SRC_DIR)/exe

# Find all .cu files in cuda/ directory
SRC_FILES := $(wildcard $(SRC_DIR)/*.cu)
# Generate corresponding executable names in cuda/exe/
EXE_FILES := $(patsubst $(SRC_DIR)/%.cu, $(EXE_DIR)/%, $(SRC_FILES))

all: $(EXE_FILES)

$(EXE_DIR)/%: $(SRC_DIR)/%.cu | $(EXE_DIR)
	echo "Compiling $< to $@"
	$(NVCC) $(CFLAGS) $< -o $@

$(EXE_DIR):
	echo "Creating output directory $(EXE_DIR) if not exists..."
	mkdir -p $(EXE_DIR)

clean:
	echo "Cleaning up generated files..."
	rm -rf $(EXE_DIR)

.PHONY: all clean
