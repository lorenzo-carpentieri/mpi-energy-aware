NVCC = nvcc
CFLAGS = -lmpi -lnvidia-ml -lnccl -DNVML -I/leonardo/home/userexternal/lcarpent/mpi-energy/mpi-energy-aware/cuda/power-profiler
SRC_DIR = $(CURDIR)
EXE_DIR = $(SRC_DIR)/exe
LOG_DIR = $(CURDIR)/../logs

# Find all .cu files in cuda/ directory
SRC_FILES := $(wildcard $(SRC_DIR)/*.cu)
# Generate corresponding executable names in cuda/exe/
EXE_FILES := $(patsubst $(SRC_DIR)/%.cu, $(EXE_DIR)/%, $(SRC_FILES))
# Generate corresponding log directories
LOG_DIRS := $(patsubst $(SRC_DIR)/%.cu, $(LOG_DIR)/%, $(SRC_FILES))

all: $(EXE_FILES) $(LOG_DIRS)

$(EXE_DIR)/%: $(SRC_DIR)/%.cu | $(EXE_DIR) $(LOG_DIR)/%
	@echo "Compiling $< to $@"
	$(NVCC) $(CFLAGS) $< -o $@

$(EXE_DIR):
	@echo "Creating output directory $(EXE_DIR) if not exists..."
	mkdir -p $(EXE_DIR)

$(LOG_DIR)/%:
	@echo "Creating log directory $@"
	mkdir -p $@

clean:
	@echo "Cleaning up generated files..."
	rm -rf $(EXE_DIR) $(LOG_DIR)

.PHONY: all clean
