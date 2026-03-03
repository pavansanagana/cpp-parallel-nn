#include <iostream>
#include <cstdlib>
#include "nn.hpp"
#include "timer.hpp"

int main() {
    // Simple 2-class blobs -> 2 outputs
    Tensor2D X;
    std::vector<int> y;
    make_blobs(X, y, 200000, 123);

    // Train/val split
    size_t N = X.rows;
    size_t Ntr = (size_t)(0.8 * N);
    Tensor2D Xtr(Ntr, 2, 0.0f), Xva(N - Ntr, 2, 0.0f);
    std::vector<int> ytr(Ntr), yva(N - Ntr);

    for (size_t i = 0; i < Ntr; ++i) {
        Xtr(i,0)=X(i,0); Xtr(i,1)=X(i,1); ytr[i]=y[i];
    }
    for (size_t i = Ntr; i < N; ++i) {
        size_t j = i - Ntr;
        Xva(j,0)=X(i,0); Xva(j,1)=X(i,1); yva[j]=y[i];
    }

    // MLP: 2 -> 128 -> 64 -> 2
    MLP net(2, 1024, 512, 2);
    float lr = 0.05f;
    int epochs = 15;
    int batch = 512;

    bool use_omp = true; // compare by toggling this
    std::cout << "use_omp=" << (use_omp ? "true" : "false") << "\n";

    for (int ep = 1; ep <= epochs; ++ep) {
        Timer t;
        // Mini-batch SGD
        for (size_t start = 0; start < Ntr; start += (size_t)batch) {
            size_t end = std::min(start + (size_t)batch, Ntr);
            size_t B = end - start;

            Tensor2D Xb(B, 2, 0.0f);
            std::vector<int> yb(B);

            for (size_t i = 0; i < B; ++i) {
                Xb(i,0)=Xtr(start+i,0);
                Xb(i,1)=Xtr(start+i,1);
                yb[i]=ytr[start+i];
            }

            net.forward(Xb, use_omp);

            Tensor2D dW1, dW2, dW3;
            std::vector<float> db1, db2, db3;
            net.backward(Xb, yb, use_omp, dW1, db1, dW2, db2, dW3, db3);
            net.sgd(lr, dW1, db1, dW2, db2, dW3, db3);
        }

        // Eval
        auto Ptr = net.forward(Xtr, use_omp);
        float Ltr = net.loss(ytr);
        float Atr = accuracy(Ptr, ytr);

        auto Pva = net.forward(Xva, use_omp);
        float Lva = net.loss(yva);
        float Ava = accuracy(Pva, yva);

        std::cout << "Epoch " << ep
                  << " time=" << t.ms() << "ms"
                  << " train_loss=" << Ltr << " train_acc=" << Atr
                  << " val_loss=" << Lva << " val_acc=" << Ava
                  << "\n";
    }
    return 0;
}
