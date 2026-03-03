#pragma once
#include <chrono>

struct Timer {
    using clock = std::chrono::steady_clock;
    clock::time_point t0;
    Timer() : t0(clock::now()) {}
    double ms() const {
        auto t1 = clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};