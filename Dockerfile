# 昇腾训练/量化/转换环境：固定 CANN 版本，保证"一条命令可复现"。
# 基础镜像以华为官方 CANN 镜像为准（版本号按你的实测芯片固定）。
# 示例：310B4 -> CANN 8.0.RC1。
FROM ascendhub.huawei.com/public-ascendhub/cann:8.0.RC1-910b-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git && rm -rf /var/lib/apt/lists/*

# 训练/量化依赖（仅训练机需要 torch/ultralytics；amct 随 CANN 或单独安装）
RUN pip3 install --no-cache-dir numpy opencv-python matplotlib \
    torch ultralytics amct_onnx amct_pytorch

WORKDIR /workspace/orchard310b
COPY . .

# 复现入口：
#   python scripts/run_demo.py          # 无硬件自测
#   bash deploy/atc/build_om.sh         # ONNX -> OM
#   bash quant/run_ptq.sh               # PTQ -> 敏感度 -> 混合精度
CMD ["bash"]
