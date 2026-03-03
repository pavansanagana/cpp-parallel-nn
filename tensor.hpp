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