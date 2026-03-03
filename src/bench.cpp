#include <iostream>
#include <chrono>
#include "include/tensor.hpp"

static double time_it(void (*fn)(const Tensor2D&, const Tensor2D&, Tensor2D&),
                      const Tensor2D& A, const Tensor2D& B, Tensor2D& C) {
    auto start = std::chrono::high_resolution_clock::now();
    fn(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

int main() {
    const int N = 512;

    Tensor2D A(N, N), B(N, N), C(N, N);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A(i, j) = 1.0;
            B(i, j) = 1.0;
            C(i, j) = 0.0;
        }
    }

    // Warmup
    matmul2d_naive(A, B, C);

    double t_naive = time_it(matmul2d_naive, A, B, C);
    double t_omp   = time_it(matmul2d_omp,   A, B, C);

    std::cout << "Matrix size: " << N << "x" << N << "\n";
    std::cout << "CPU naive:   " << t_naive << " s\n";
    std::cout << "CPU OpenMP:  " << t_omp   << " s\n";
    std::cout << "Speedup:     " << (t_naive / t_omp) << "x\n";

#ifdef USE_CUDA
    double t_cuda = time_it(matmul2d_cuda, A, B, C);
    std::cout << "CUDA:        " << t_cuda << " s\n";
    std::cout << "GPU speedup: " << (t_naive / t_cuda) << "x\n";
#endif

    return 0;
}
