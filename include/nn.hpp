#pragma once
#include "tensor.hpp"
#include <vector>

Tensor2D add_bias(const Tensor2D& X, const std::vector<float>& b);
Tensor2D relu(const Tensor2D& X);
Tensor2D relu_backward(const Tensor2D& dY, const Tensor2D& Y); // Y = relu(X)
Tensor2D softmax(const Tensor2D& logits);
float cross_entropy(const Tensor2D& probs, const std::vector<int>& y);

struct MLP {
    // Shapes:
    // X: (B, Din)
    // W1: (Din, H1), b1: (H1)
    // W2: (H1, H2), b2: (H2)
    // W3: (H2, C),  b3: (C)
    Tensor2D W1, W2, W3;
    std::vector<float> b1, b2, b3;

    // caches
    Tensor2D Z1, A1, Z2, A2, Z3, P;

    MLP(size_t Din, size_t H1, size_t H2, size_t C);

    Tensor2D forward(const Tensor2D& X, bool use_omp);
    float loss(const std::vector<int>& y);
    void backward(const Tensor2D& X, const std::vector<int>& y, bool use_omp,
                  Tensor2D& dW1, std::vector<float>& db1,
                  Tensor2D& dW2, std::vector<float>& db2,
                  Tensor2D& dW3, std::vector<float>& db3);

    void sgd(float lr,
             const Tensor2D& dW1, const std::vector<float>& db1,
             const Tensor2D& dW2, const std::vector<float>& db2,
             const Tensor2D& dW3, const std::vector<float>& db3);
};

void make_blobs(Tensor2D& X, std::vector<int>& y, size_t N, uint32_t seed=123);
float accuracy(const Tensor2D& probs, const std::vector<int>& y);
