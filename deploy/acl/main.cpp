// ACL 推理主程序骨架：三线程流水入口。
#include <cstdio>

#include "pipeline.hpp"

// 前向声明（后处理在 deploy/postprocess 下，链接时合并）
#include "../postprocess/nms.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.om>\n", argv[0]);
        return 1;
    }
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtRunMode run_mode;
    aclrtGetRunMode(&run_mode);  // ACL_DEVICE / ACL_HOST

    orchard::DeviceBuffers buf;
    if (!buf.init(argv[1])) {
        std::fprintf(stderr, "load om failed\n");
        return 2;
    }
    std::printf("model loaded, input=%zu B output=%zu B\n",
                buf.input_size, buf.output_size);
    // 这里接入三线程流水（T1 取帧 / T2 预处理+推理 / T3 后处理）与
    // deploy/postprocess 的 NMS。见 docs/ 与 README 的完整接线说明。
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
