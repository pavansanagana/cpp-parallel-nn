#pragma once

#include <vector>
#include <cassert>

// Reuse your existing Tensor2D definition if it's in root tensor.hpp
// (Your project already has tensor.hpp and it built successfully.)
#include "../tensor.hpp"

inline void matmul2d_naive(const Tensor2D& A, const Tensor2D& B, Tensor2D& C) {
    const int N = A.rows;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
}

inline void matmul2d_omp(const Tensor2D& A, const Tensor2D& B, Tensor2D& C) {
    const int N = A.rows;

#if defined(HAS_OPENMP)
  #pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
}

// Keep your old name working (calls naive by default)
inline void matmul2d(const Tensor2D& A, const Tensor2D& B, Tensor2D& C) {
    matmul2d_naive(A, B, C);
}

#ifdef USE_CUDA
// Implemented in cuda/matmul.cu (only builds when USE_CUDA=ON + nvcc exists)
void matmul2d_cuda(const Tensor2D& A, const Tensor2D& B, Tensor2D& C);
#endif
