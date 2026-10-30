// NMS 基准：8400 随机候选框，打印耗时与保留数。
// 编译：g++ -O2 -std=c++11 bench_nms.cpp nms.cpp -o bench_nms
#include <chrono>
#include <cstdio>
#include <random>

#include "nms.hpp"

int main() {
    const int n = 8400;
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> u(0.f, 640.f);
    std::uniform_real_distribution<float> w(1.f, 60.f);
    std::uniform_real_distribution<float> s(0.f, 1.f);
    std::vector<float> boxes(4 * n), scores(n);
    std::vector<int> cls(n);
    for (int i = 0; i < n; ++i) {
        boxes[4 * i + 0] = u(rng); boxes[4 * i + 1] = u(rng);
        boxes[4 * i + 2] = boxes[4 * i + 0] + w(rng);
        boxes[4 * i + 3] = boxes[4 * i + 1] + w(rng);
        scores[i] = s(rng);
        cls[i] = i % 4;
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    auto keep = orchard::nms(boxes, scores, 0.45f, 300, &cls);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("boxes=%d kept=%zu time=%.3f ms\n", n, keep.size(), ms);
    return 0;
}
