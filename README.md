# C++ Matrix Multiplication Benchmark

This project benchmarks different implementations of matrix multiplication.

## Implementations

1. CPU naive implementation
2. CPU OpenMP parallel implementation
3. CUDA GPU implementation (optional)

## Build

CPU version:

brew install llvm libomp cmake && rm -rf build && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$(brew --prefix llvm)/bin/clang" -DCMAKE_CXX_COMPILER="$(brew --prefix llvm)/bin/clang++" && cmake --build build -j && ./build/bench
