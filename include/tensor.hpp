#pragma once
#include <vector>
#include <cstddef>

struct Tensor2D {
    size_t rows = 0, cols = 0;
    std::vector<float> a;

    Tensor2D() = default;
    Tensor2D(size_t r, size_t c, float init = 0.0f) : rows(r), cols(c), a(r*c, init) {}

    inline float& operator()(size_t r, size_t c) {
        return a[r * cols + c];
    }
    inline const float& operator()(size_t r, size_t c) const {
        return a[r * cols + c];
    }
};

inline void matmul2d(const Tensor2D& A, const Tensor2D& B, Tensor2D& C) {
    // assumes square matrices NxN
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
