# Makefile for casm_bfCorr, the FRB Search Pipeline, and casm_capture

# --- Compilers ---
NVCC := /usr/bin/nvcc
CC   := gcc

# --- Common Flags and Paths ---
COMMON_CFLAGS   := -g -O3 -std=c++14 -arch=sm_75 -Xcompiler="-pthread"
COMMON_INCLUDES := -I/usr/local/cuda/include -I/usr/local/include/psrdada
COMMON_LIBPATHS := -L/usr/local/cuda/lib64 -L/usr/local/lib

# --- casm_capture (DADA capture) ---
TARGET_CAPTURE   := casm_capture
SRC_CAPTURE      := casm_capture.c
CAPTURE_CFLAGS   := -Wall -O2
CAPTURE_INCLUDES := $(COMMON_INCLUDES)
CAPTURE_LIBPATHS := $(COMMON_LIBPATHS)
# Link against psrdada (which provides the DADA client API), pthread, and math
LIBS_CAPTURE     := -lpsrdada -lpthread -lm

# --- Project 1: casm_bfCorr ---
TARGET_BFCORR := casm_bfCorr
SRC_BFCORR    := casm_bfCorr.cu
LIBS_BFCORR   := -lpsrdada -lcublas

# --- Project 2: frb_search_pipeline ---
TARGET_PIPELINE := casm_hella
SRC_PIPELINE    := casm_hella.cu
DEDISP_INCLUDES := -I/usr/local/include
DEDISP_LIBPATHS := -L/usr/local/lib
LIBS_PIPELINE   := -lpsrdada -lcublas -ldedisp -lnppc -lnppi -lnpps -lcurand

# --- Build Rules ---

# Default: build all executables
all: $(TARGET_BFCORR) $(TARGET_PIPELINE) $(TARGET_CAPTURE)

# Build casm_bfCorr
$(TARGET_BFCORR): $(SRC_BFCORR) casm_def.h
	@echo "--- Compiling $(TARGET_BFCORR) ---"
	$(NVCC) -o $@ $< $(COMMON_CFLAGS) $(COMMON_INCLUDES) \
           $(COMMON_LIBPATHS) $(LIBS_BFCORR)

# Build frb_search_pipeline
$(TARGET_PIPELINE): $(SRC_PIPELINE) casm_def.h
	@echo "--- Compiling $(TARGET_PIPELINE) ---"
	$(NVCC) -o $@ $< $(COMMON_CFLAGS) $(COMMON_INCLUDES) \
           $(DEDISP_INCLUDES) $(COMMON_LIBPATHS) \
           $(DEDISP_LIBPATHS) $(LIBS_PIPELINE)

# Build casm_capture
$(TARGET_CAPTURE): $(SRC_CAPTURE)
	@echo "--- Compiling $(TARGET_CAPTURE) ---"
	$(CC) $(CAPTURE_CFLAGS) \
	    -o $@ $< \
	    $(CAPTURE_INCLUDES) $(CAPTURE_LIBPATHS) $(LIBS_CAPTURE)

# Clean up all targets
clean:
	@echo "--- Cleaning up all targets ---"
	rm -f $(TARGET_BFCORR) $(TARGET_PIPELINE) $(TARGET_CAPTURE)

.PHONY: all clean
