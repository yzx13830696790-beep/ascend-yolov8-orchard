// 向量化 NMS 实现。算法：分数降序 + 贪心抑制 + class-offset。
#include "nms.hpp"

#include <algorithm>
#include <numeric>

namespace orchard {

static inline float iou(const float* a, const float* b) {
    float xx1 = a[0] > b[0] ? a[0] : b[0];
    float yy1 = a[1] > b[1] ? a[1] : b[1];
    float xx2 = a[2] < b[2] ? a[2] : b[2];
    float yy2 = a[3] < b[3] ? a[3] : b[3];
    float w = xx2 - xx1; if (w < 0) w = 0;
    float h = yy2 - yy1; if (h < 0) h = 0;
    float inter = w * h;
    float area_a = (a[2] - a[0]) * (a[3] - a[1]);
    float area_b = (b[2] - b[0]) * (b[3] - b[1]);
    float uni = area_a + area_b - inter;
    return inter / (uni + 1e-9f);
}

std::vector<int> nms(const std::vector<float>& boxes,
                     const std::vector<float>& scores,
                     float iou_thr, int max_det,
                     const std::vector<int>* class_ids) {
    const int n = static_cast<int>(scores.size());
    std::vector<int> idx(n), keep;
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return scores[a] > scores[b];
    });

    // class-offset 平移量
    float max_coord = 0.f;
    for (int i = 0; i < n; ++i)
        max_coord = std::max(max_coord, boxes[4 * i + 2]);
    std::vector<float> offset(n, 0.f);
    if (class_ids)
        for (int i = 0; i < n; ++i)
            offset[i] = static_cast<float>((*class_ids)[i]) * (max_coord + 1.f);

    keep.reserve(max_det);
    std::vector<char> suppressed(n, 0);
    for (int order_i = 0; order_i < n; ++order_i) {
        int i = idx[order_i];
        if (suppressed[i]) continue;
        keep.push_back(i);
        if (static_cast<int>(keep.size()) >= max_det) break;
        const float* a = &boxes[4 * i];
        const float oa = offset[i];
        for (int order_j = order_i + 1; order_j < n; ++order_j) {
            int j = idx[order_j];
            if (suppressed[j]) continue;
            if (class_ids && offset[j] != oa) continue;  // 不同类必不抑制
            const float* b = &boxes[4 * j];
            if (iou(a, b) > iou_thr) suppressed[j] = 1;
        }
    }
    return keep;
}

}  // namespace orchard
