# Makefile for casm_bfCorr.cu

# Compiler and flags
NVCC := /usr/bin/nvcc
CFLAGS := -O3 -std=c++14 -arch=sm_75 -Xcompiler="-pthread"

# Include and library paths (customize if needed)
INCLUDES := -I/usr/local/cuda/include
LIBPATHS := -L/usr/local/cuda/lib64 -L/usr/local/lib

# Libraries
LIBS := -lcufft -lcublas -lpsrdada

# Target and sources
TARGET := casm_bfCorr
SRC := casm_bfCorr.cu

# Default rule
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -o $@ $^ $(CFLAGS) $(INCLUDES) $(LIBPATHS) $(LIBS)

# Clean rule
clean:
	rm -f $(TARGET)
