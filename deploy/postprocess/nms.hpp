// 向量化（单遍贪心 + class-offset）NMS，8400 候选框目标 <1ms。
// 与 orch/nms.py 的算法等价；多类用 class_id * offset 平移，避免按类循环。
#pragma once

#include <vector>

namespace orchard {

struct Box {
    float x1, y1, x2, y2;
};

// boxes: x1,y1,x2,y2 连续排布；scores 同序；class_ids 可为 nullptr（单类）。
// 返回保留框的原始下标，按分数降序。
std::vector<int> nms(const std::vector<float>& boxes,
                     const std::vector<float>& scores,
                     float iou_thr, int max_det,
                     const std::vector<int>* class_ids = nullptr);

}  // namespace orchard
