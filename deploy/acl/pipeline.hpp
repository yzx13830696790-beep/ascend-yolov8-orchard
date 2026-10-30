// ACL 三线程流水 + 双 buffer + Device 常驻内存。
//
// 三个线程（对应规划书"线 B · 推理工程"）：
//   T1 取帧：      摄像头/DVPP 解码 -> 入队(双 buffer 轮换)
//   T2 预处理+推理：Host 预处理 + aclmdlExecute（Device 常驻输入/输出）
//   T3 后处理：     解码 + 向量化 NMS + 坐标还原
//
// 说明：本文件是目标机（昇腾 Atlas）上的工程骨架，需在装有 CANN 的
// 环境用 CMake 编译；本机无 CANN，未编译，接口按 ACL 官方 API 书写。
#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

#include "acl/acl.h"

namespace orchard {

// 双 buffer：Device 侧输入输出只申请一次，避免每帧 H2D/D2H 大块拷贝
struct DeviceBuffers {
    void* input_dev = nullptr;    // aclrtMalloc
    void* output_dev = nullptr;
    size_t input_size = 0;
    size_t output_size = 0;
    uint32_t model_id = 0;
    aclmdlDesc* desc = nullptr;
    aclmdlDataset* in_ds = nullptr;
    aclmdlDataset* out_ds = nullptr;

    bool init(const char* om_path) {
        aclError ret = aclmdlLoadFromFile(om_path, &model_id);
        if (ret != ACL_SUCCESS) return false;
        desc = aclmdlCreateDesc();
        aclmdlGetDesc(desc, model_id);
        input_size = aclmdlGetInputSizeByIndex(desc, 0);
        output_size = aclmdlGetOutputSizeByIndex(desc, 0);
        aclrtMalloc(&input_dev, input_size, ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&output_dev, output_size, ACL_MEM_MALLOC_HUGE_FIRST);
        in_ds = aclmdlCreateDataset();
        out_ds = aclmdlCreateDataset();
        aclDataBuffer* in_buf = aclCreateDataBuffer(input_dev, input_size);
        aclDataBuffer* out_buf = aclCreateDataBuffer(output_dev, output_size);
        aclmdlAddDatasetBuffer(in_ds, in_buf);
        aclmdlAddDatasetBuffer(out_ds, out_buf);
        return true;
    }

    bool infer(const void* host_in, void* host_out) {
        aclrtMemcpy(input_dev, input_size, host_in, input_size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
        aclmdlExecute(model_id, in_ds, out_ds);
        aclrtMemcpy(host_out, output_size, output_dev, output_size,
                    ACL_MEMCPY_DEVICE_TO_HOST);
        return true;
    }
};

// 有界队列：线程间传递帧（双 buffer 轮换）
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t cap) : cap_(cap) {}
    void push(T v) {
        std::unique_lock<std::mutex> lk(m_);
        cv_full_.wait(lk, [&] { return q_.size() < cap_; });
        q_.push(std::move(v));
        cv_empty_.notify_one();
    }
    T pop() {
        std::unique_lock<std::mutex> lk(m_);
        cv_empty_.wait(lk, [&] { return !q_.empty(); });
        T v = std::move(q_.front());
        q_.pop();
        cv_full_.notify_one();
        return v;
    }
private:
    size_t cap_;
    std::queue<T> q_;
    std::mutex m_;
    std::condition_variable cv_full_, cv_empty_;
};

}  // namespace orchard
