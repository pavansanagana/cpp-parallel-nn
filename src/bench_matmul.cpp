#include <iostream>
#include <random>
#include <cmath>
#include <cstdlib>
#include <omp.h>

#include "tensor.hpp"
#include "matmul.hpp"
#include "timer.hpp"

static void fill_rand(Tensor2D& T, uint32_t seed = 123) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : T.a) x = dist(rng);
}

static float checksum(const Tensor2D& T) {
    // Prevent optimizer from discarding work
    double s = 0.0;
    for (float x : T.a) s += x;
    return (float)s;
}

int main(int argc, char** argv) {
    // Example sizes similar to MLP layer matmuls: (batch x in) * (in x out)
    size_t M = 2048, K = 512, N = 512;
    int iters = 5;

    if (argc >= 4) {
        M = (size_t)std::stoul(argv[1]);
        K = (size_t)std::stoul(argv[2]);
        N = (size_t)std::stoul(argv[3]);
    }
    if (argc >= 5) iters = std::stoi(argv[4]);

    Tensor2D A(M, K), B(K, N);
    fill_rand(A, 1);
    fill_rand(B, 2);

    std::cout << "Matrix sizes: A(" << M << "," << K << ") * B(" << K << "," << N << ")\n";
    std::cout << "OpenMP max threads: " << omp_get_max_threads() << "\n\n";

    // Warmup
    auto Cw = matmul_serial(A, B);
    std::cout << "Warmup checksum: " << checksum(Cw) << "\n\n";

    // Serial timing
    double best_serial = 1e18;
    float cs_serial = 0.0f;
    for (int t = 0; t < iters; ++t) {
        Timer tm;
        auto C = matmul_serial(A, B);
        double ms = tm.ms();
        best_serial = std::min(best_serial, ms);
        cs_serial = checksum(C);
    }

    // OMP timing
    double best_omp = 1e18;
    float cs_omp = 0.0f;
    for (int t = 0; t < iters; ++t) {
        Timer tm;
        auto C = matmul_omp(A, B);
        double ms = tm.ms();
        best_omp = std::min(best_omp, ms);
        cs_omp = checksum(C);
    }

    std::cout << "Best serial: " << best_serial << " ms\n";
    std::cout << "Best omp   : " << best_omp << " ms\n";
    std::cout << "Speedup    : " << (best_serial / best_omp) << "x\n";
    std::cout << "Checksums  : serial=" << cs_serial << " omp=" << cs_omp << "\n";

    // Quick correctness sanity (allow tiny float diff)
    float diff = std::fabs(cs_serial - cs_omp);
    std::cout << "Checksum diff: " << diff << "\n";
    return 0;
}