# CASM-XEngine: A GPU-Accelerated Correlator and Beamformer

This repository contains the source code for `casm-xengine`, a high-performance correlator and beamformer designed for the Coherent All-Sky Monitor based on Vikram Ravi's dsaX_xengine software that was developed fro the DSA-110. It utilizes CUDA for GPU acceleration to process data from antenna arrays.

## Project Purpose

The primary function of this software is to capture, correlate, and beamform astronomical signals from a radio telescope. It appears to be designed for use with systems like DSA-X.

## Key Components

The core logic is implemented in C and CUDA C++, located in the `src/` directory. Key files include:
-   `casm_correlator.c`, `casm_correlator_2.c`: Main correlator logic.
-   `casm_bfCorr.cu`, `casm_hella.cu`: GPU-accelerated beamforming and correlation kernels.
-   `casm_capture.c`, `dsaX_capture.c`: Data capture modules.
-   `read_vis.py`: A Python script for reading visibility data, likely for analysis or debugging.

## Dependencies

-   **Python 3.10+**
-   **blimpy**: For handling Breakthrough Listen data formats.
-   **CUDA Toolkit**: For compiling the `.cu` source files.

## Building

The project uses a `Makefile` for compilation. To build the project, you will likely need to run:

```bash
make
```

in the `src/` directory. Please refer to the `Makefile` for specific targets and options.

## Usage

The `scripts/` directory contains scripts for running the data processing pipeline, such as `run_pipeline.py` and `pipeline.sh`. These scripts likely use the compiled executables from `src/`.
