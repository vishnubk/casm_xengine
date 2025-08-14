# CASM-XEngine: GPU-Accelerated Radio Astronomy Correlator & Beamformer

High-performance CUDA-based correlator and beamformer for radio astronomy, derived from the DSA-110 xengine. Optimized for real-time processing of multi-beam, multi-antenna data streams.

## Features

- **Real-time beamforming**: 256+ antennas × 1024+ beams
- **GPU acceleration**: CUDA kernels for CUBLAS operations
- **Performance monitoring**: Built-in benchmarking with real-time ratio analysis
- **Auto-configuration**: Automatically reads parameters from `casm_def.h`
- **DADA integration**: PSRDADA-compatible data handling

## Quick Start

### Build
```bash
cd src
make clean && make
```

### Benchmark
```bash
cd tests
python benchmark_bfCorr.py
```

### Run Pipeline
```bash
cd scripts
./pipeline.sh
```

## Configuration

Key parameters in `src/casm_def.h`:
- `NBEAMS`: Number of beamforming directions (default: 256)
- `NANTS`: Number of antennas (default: 256)
- `NPACKETS_PER_BLOCK`: Data block size (default: 2048)

## Performance

The benchmark system provides:
- Real-time performance ratios
- GPU timing breakdown (copy, prep, CUBLAS, output)
- Throughput metrics (beams/second, operations/second)
- Automatic timeout and process management

## Dependencies

- CUDA Toolkit 11+
- PSRDADA library
- Python 3.8+ (for benchmarking)
- GCC/Clang for C compilation

## Architecture

- **`casm_bfCorr.cu`**: Main beamformer with GPU kernels
- **`casm_capture.c`**: Data capture from antenna arrays
- **`fake_writer`**: Test data generator for development
- **`benchmark_bfCorr.py`**: Performance testing framework

## License

Derived from DSA-110 xengine (see original: https://github.com/dsa110/dsa110-xengine)
