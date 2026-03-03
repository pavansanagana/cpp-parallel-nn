# cpp-parallel-nn

Performance-focused C++ mini project for experimenting with tensor ops + CPU parallelism (and later CUDA).
Goal: benchmark matrix multiplication and show clear single-thread vs parallel speedups.

## Build
```bash
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
