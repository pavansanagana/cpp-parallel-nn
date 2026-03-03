#include "matmul.hpp"
#include <stdexcept>
#include <omp.h>
#include <algorithm>

static inline void check_dims(const Tensor2D& A, const Tensor2D& B) {
    if (A.cols != B.rows) throw std::runtime_error("matmul dim mismatch: A.cols must equal B.rows");
}

Tensor2D matmul_serial(const Tensor2D& A, const Tensor2D& B) {
    check_dims(A, B);
    Tensor2D C(A.rows, B.cols, 0.0f);

    for (size_t i = 0; i < A.rows; ++i) {
        for (size_t k = 0; k < A.cols; ++k) {
            float aik = A(i, k);
            const float* b_row = &B.a[k * B.cols];
            float* c_row = &C.a[i * C.cols];
            for (size_t j = 0; j < B.cols; ++j) {
                c_row[j] += aik * b_row[j];
            }
        }
    }
    return C;
}

Tensor2D matmul_omp(const Tensor2D& A, const Tensor2D& B) {
    check_dims(A, B);
    Tensor2D C(A.rows, B.cols, 0.0f);

    #pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)A.rows; ++i) {
        for (size_t k = 0; k < A.cols; ++k) {
            float aik = A((size_t)i, k);
            const float* b_row = &B.a[k * B.cols];
            float* c_row = &C.a[(size_t)i * C.cols];
            for (size_t j = 0; j < B.cols; ++j) {
                c_row[j] += aik * b_row[j];
            }
        }
    }
    return C;
}

// Blocked (tiled) OpenMP matmul for better cache locality
static Tensor2D matmul_omp_blocked_impl(const Tensor2D& A, const Tensor2D& B, int BS) {
    check_dims(A, B);
    Tensor2D C(A.rows, B.cols, 0.0f);

    const int M = (int)A.rows;
    const int K = (int)A.cols;
    const int N = (int)B.cols;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < M; ii += BS) {
        for (int jj = 0; jj < N; jj += BS) {
            const int i_max = std::min(ii + BS, M);
            const int j_max = std::min(jj + BS, N);

            for (int kk = 0; kk < K; kk += BS) {
                const int k_max = std::min(kk + BS, K);

                for (int i = ii; i < i_max; ++i) {
                    float* c_row = &C.a[(size_t)i * (size_t)N];
                    for (int k = kk; k < k_max; ++k) {
                        const float aik = A.a[(size_t)i * (size_t)K + (size_t)k];
                        const float* b_row = &B.a[(size_t)k * (size_t)N];
                        for (int j = jj; j < j_max; ++j) {
                            c_row[j] += aik * b_row[j];
                        }
                    }
                }
            }
        }
    }
    return C;
}

Tensor2D matmul_omp_blocked(const Tensor2D& A, const Tensor2D& B) {
    return matmul_omp_blocked_impl(A, B, 32);
}
