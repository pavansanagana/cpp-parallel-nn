#include "../include/tensor.hpp"

// Placeholder: real CUDA kernel will go here later.
// For now, just call CPU OpenMP so linking works when CUDA builds.
void matmul2d_cuda(const Tensor2D& A, const Tensor2D& B, Tensor2D& C) {
    matmul2d_omp(A, B, C);
}
