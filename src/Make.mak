################################################################################
#  Minimal Makefile for CASM-258 beamformer
################################################################################

CUDA_PATH  ?= /usr/local/cuda
NVCC       ?= $(CUDA_PATH)/bin/nvcc

GPU_ARCH   ?= sm_90        # change to sm_80, sm_70, etc. if needed
CFLAGS     ?= -O3 -std=c++17
GENCODE    ?= -gencode arch=compute_$(GPU_ARCH:sm_%=%),code=$(GPU_ARCH)

TARGET     := casm_bfCorr
SRCS       := casm_bfCorr_CASM258.cu

$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) $(GENCODE) -o $@ $^

clean:
	rm -f $(TARGET) *.o