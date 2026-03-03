#include "nn.hpp"
#include "matmul.hpp"
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>

static void randn_fill(Tensor2D& T, float scale, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : T.a) v = dist(rng) * scale;
}

static std::vector<float> zeros(size_t n) { return std::vector<float>(n, 0.0f); }

Tensor2D add_bias(const Tensor2D& X, const std::vector<float>& b) {
    if (X.cols != b.size()) throw std::runtime_error("add_bias mismatch");
    Tensor2D Y(X.rows, X.cols, 0.0f);
    for (size_t i = 0; i < X.rows; ++i) {
        for (size_t j = 0; j < X.cols; ++j) {
            Y(i,j) = X(i,j) + b[j];
        }
    }
    return Y;
}

Tensor2D relu(const Tensor2D& X) {
    Tensor2D Y(X.rows, X.cols, 0.0f);
    for (size_t i = 0; i < X.a.size(); ++i) Y.a[i] = (X.a[i] > 0.0f) ? X.a[i] : 0.0f;
    return Y;
}

Tensor2D relu_backward(const Tensor2D& dY, const Tensor2D& Y) {
    Tensor2D dX(dY.rows, dY.cols, 0.0f);
    for (size_t i = 0; i < dY.a.size(); ++i) dX.a[i] = (Y.a[i] > 0.0f) ? dY.a[i] : 0.0f;
    return dX;
}

Tensor2D softmax(const Tensor2D& logits) {
    Tensor2D P(logits.rows, logits.cols, 0.0f);
    for (size_t i = 0; i < logits.rows; ++i) {
        float mx = -1e30f;
        for (size_t j = 0; j < logits.cols; ++j) mx = std::max(mx, logits(i,j));
        float sum = 0.0f;
        for (size_t j = 0; j < logits.cols; ++j) {
            float e = std::exp(logits(i,j) - mx);
            P(i,j) = e;
            sum += e;
        }
        float inv = 1.0f / sum;
        for (size_t j = 0; j < logits.cols; ++j) P(i,j) *= inv;
    }
    return P;
}

float cross_entropy(const Tensor2D& probs, const std::vector<int>& y) {
    const float eps = 1e-12f;
    float L = 0.0f;
    for (size_t i = 0; i < probs.rows; ++i) {
        int yi = y[i];
        float p = probs(i, (size_t)yi);
        L += -std::log(std::max(p, eps));
    }
    return L / (float)probs.rows;
}

MLP::MLP(size_t Din, size_t H1, size_t H2, size_t C)
: W1(Din, H1), W2(H1, H2), W3(H2, C),
  b1(zeros(H1)), b2(zeros(H2)), b3(zeros(C))
{
    // He init for ReLU layers
    randn_fill(W1, std::sqrt(2.0f / (float)Din), 1);
    randn_fill(W2, std::sqrt(2.0f / (float)H1), 2);
    randn_fill(W3, std::sqrt(2.0f / (float)H2), 3);
}

Tensor2D MLP::forward(const Tensor2D& X, bool use_omp) {
    // Z1 = XW1 + b1
    Z1 = use_omp ? matmul_omp_blocked(X, W1) : matmul_serial(X, W1);
    Z1 = add_bias(Z1, b1);
    A1 = relu(Z1);

    Z2 = use_omp ? matmul_omp_blocked(A1, W2) : matmul_serial(A1, W2);
    Z2 = add_bias(Z2, b2);
    A2 = relu(Z2);

    Z3 = use_omp ? matmul_omp_blocked(A2, W3) : matmul_serial(A2, W3);
    Z3 = add_bias(Z3, b3);

    P = softmax(Z3);
    return P;
}

float MLP::loss(const std::vector<int>& y) {
    return cross_entropy(P, y);
}

static Tensor2D transpose(const Tensor2D& X) {
    Tensor2D T(X.cols, X.rows, 0.0f);
    for (size_t i = 0; i < X.rows; ++i)
        for (size_t j = 0; j < X.cols; ++j)
            T(j,i) = X(i,j);
    return T;
}

static std::vector<float> col_sum(const Tensor2D& X) {
    std::vector<float> s(X.cols, 0.0f);
    for (size_t i = 0; i < X.rows; ++i)
        for (size_t j = 0; j < X.cols; ++j)
            s[j] += X(i,j);
    return s;
}

void MLP::backward(const Tensor2D& X, const std::vector<int>& y, bool use_omp,
                   Tensor2D& dW1, std::vector<float>& db1o,
                   Tensor2D& dW2, std::vector<float>& db2o,
                   Tensor2D& dW3, std::vector<float>& db3o)
{
    size_t B = X.rows;
    // dZ3 = P; dZ3[i, y[i]] -= 1; then /B
    Tensor2D dZ3 = P;
    for (size_t i = 0; i < B; ++i) dZ3(i, (size_t)y[i]) -= 1.0f;
    for (auto& v : dZ3.a) v /= (float)B;

    // dW3 = A2^T dZ3
    Tensor2D A2T = transpose(A2);
    dW3 = use_omp ? matmul_omp_blocked(A2T, dZ3) : matmul_serial(A2T, dZ3);
    db3o = col_sum(dZ3);

    // dA2 = dZ3 W3^T
    Tensor2D W3T = transpose(W3);
    Tensor2D dA2 = use_omp ? matmul_omp_blocked(dZ3, W3T) : matmul_serial(dZ3, W3T);
    Tensor2D dZ2 = relu_backward(dA2, A2);

    Tensor2D A1T = transpose(A1);
    dW2 = use_omp ? matmul_omp_blocked(A1T, dZ2) : matmul_serial(A1T, dZ2);
    db2o = col_sum(dZ2);

    Tensor2D W2T = transpose(W2);
    Tensor2D dA1 = use_omp ? matmul_omp_blocked(dZ2, W2T) : matmul_serial(dZ2, W2T);
    Tensor2D dZ1 = relu_backward(dA1, A1);

    Tensor2D XT = transpose(X);
    dW1 = use_omp ? matmul_omp_blocked(XT, dZ1) : matmul_serial(XT, dZ1);
    db1o = col_sum(dZ1);
}

void MLP::sgd(float lr,
              const Tensor2D& dW1, const std::vector<float>& db1o,
              const Tensor2D& dW2, const std::vector<float>& db2o,
              const Tensor2D& dW3, const std::vector<float>& db3o)
{
    for (size_t i = 0; i < W1.a.size(); ++i) W1.a[i] -= lr * dW1.a[i];
    for (size_t i = 0; i < W2.a.size(); ++i) W2.a[i] -= lr * dW2.a[i];
    for (size_t i = 0; i < W3.a.size(); ++i) W3.a[i] -= lr * dW3.a[i];
    for (size_t i = 0; i < b1.size(); ++i) b1[i] -= lr * db1o[i];
    for (size_t i = 0; i < b2.size(); ++i) b2[i] -= lr * db2o[i];
    for (size_t i = 0; i < b3.size(); ++i) b3[i] -= lr * db3o[i];
}

void make_blobs(Tensor2D& X, std::vector<int>& y, size_t N, uint32_t seed) {
    // 2D blobs, 2 classes
    X = Tensor2D(N, 2, 0.0f);
    y.assign(N, 0);

    std::mt19937 rng(seed);
    std::normal_distribution<float> n01(0.0f, 1.0f);

    for (size_t i = 0; i < N; ++i) {
        int cls = (i < N/2) ? 0 : 1;
        y[i] = cls;
        float cx = (cls == 0) ? -2.0f : +2.0f;
        float cy = (cls == 0) ? -2.0f : +2.0f;
        X(i,0) = cx + 0.8f * n01(rng);
        X(i,1) = cy + 0.8f * n01(rng);
    }

    // Shuffle
    std::vector<size_t> idx(N);
    for (size_t i = 0; i < N; ++i) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng);

    Tensor2D X2(N, 2, 0.0f);
    std::vector<int> y2(N, 0);
    for (size_t i = 0; i < N; ++i) {
        X2(i,0) = X(idx[i],0);
        X2(i,1) = X(idx[i],1);
        y2[i] = y[idx[i]];
    }
    X = std::move(X2);
    y = std::move(y2);
}

float accuracy(const Tensor2D& probs, const std::vector<int>& y) {
    size_t correct = 0;
    for (size_t i = 0; i < probs.rows; ++i) {
        size_t arg = 0;
        float best = probs(i,0);
        for (size_t j = 1; j < probs.cols; ++j) {
            float v = probs(i,j);
            if (v > best) { best = v; arg = j; }
        }
        if ((int)arg == y[i]) correct++;
    }
    return (float)correct / (float)probs.rows;
}
