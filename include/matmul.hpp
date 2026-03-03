#pragma once
#include "tensor.hpp"

Tensor2D matmul_serial(const Tensor2D& A, const Tensor2D& B);
Tensor2D matmul_omp(const Tensor2D& A, const Tensor2D& B);
Tensor2D matmul_omp_blocked(const Tensor2D& A, const Tensor2D& B);
