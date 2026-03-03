#include <iostream>
#include <chrono>
#include "tensor.hpp"

int main() {
    const int N = 512;

    Tensor2D A(N, N);
    Tensor2D B(N, N);
    Tensor2D C(N, N);

    // Fill A and B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A(i, j) = 1.0;
            B(i, j) = 1.0;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    matmul2d(A, B, C);
    auto end = std::chrono::high_resolution_clock::now();

    double seconds = std::chrono::duration<double>(end - start).count();

    std::cout << "Matrix size: " << N << "x" << N << "\n";
    std::cout << "Execution time: " << seconds << " seconds\n";

    return 0;
}
