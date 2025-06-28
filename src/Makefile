# Makefile for casm_bfCorr and the FRB Search Pipeline (frb_search_pipeline)

# --- Compiler ---
NVCC := /usr/bin/nvcc

# --- Common Flags and Paths ---
# Flags and paths used by BOTH projects
COMMON_CFLAGS := -g -O3 -std=c++14 -arch=sm_75 -Xcompiler="-pthread"
COMMON_INCLUDES := -I/usr/local/cuda/include -I/usr/local/include/psrdada
COMMON_LIBPATHS := -L/usr/local/cuda/lib64 -L/usr/local/lib
COMMON_LIBS := -lpsrdada -lcublas

# --- Project 1: casm_bfCorr ---
TARGET_BFCORR := casm_bfCorr
SRC_BFCORR := casm_bfCorr.cu
LIBS_BFCORR := $(COMMON_LIBS) # bfCorr has no extra libraries

# --- Project 2: frb_search_pipeline ---
TARGET_PIPELINE := frb_search_pipeline
SRC_PIPELINE := pipeline.cu

# Add dedisp-specific paths and libraries for the pipeline
# <<< IMPORTANT: EDIT THESE DEDISP PATHS >>>
DEDISP_INCLUDES := -I/path/to/your/dedisp/include
DEDISP_LIBPATHS := -L/path/to/your/dedisp/lib

# Pipeline has extra libraries
LIBS_PIPELINE := $(COMMON_LIBS) -ldedisp -lnppc -lnppi -lnpps -lcurand

# --- Build Rules ---

# The default rule when you just type 'make' or 'make all'.
# It will build both executables.
all: $(TARGET_BFCORR) $(TARGET_PIPELINE)

# Rule to build casm_bfCorr
$(TARGET_BFCORR): $(SRC_BFCORR) casm_def.h
	@echo "--- Compiling $(TARGET_BFCORR) ---"
	$(NVCC) -o $@ $< $(COMMON_CFLAGS) $(COMMON_INCLUDES) $(COMMON_LIBPATHS) $(LIBS_BFCORR)

# Rule to build frb_search_pipeline
$(TARGET_PIPELINE): $(SRC_PIPELINE) casm_def.h
	@echo "--- Compiling $(TARGET_PIPELINE) ---"
	$(NVCC) -o $@ $< $(COMMON_CFLAGS) $(COMMON_INCLUDES) $(DEDISP_INCLUDES) $(COMMON_LIBPATHS) $(DEDISP_LIBPATHS) $(LIBS_PIPELINE)

# Clean rule: removes BOTH executables
clean:
	@echo "--- Cleaning up all targets ---"
	rm -f $(TARGET_BFCORR) $(TARGET_PIPELINE)

# Phony targets are not files. 'all' and 'clean' are standard phony targets.
.PHONY: all clean