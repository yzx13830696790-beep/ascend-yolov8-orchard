#!/bin/bash
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh

atc \
  --model=baseline.onnx \
  --output=baseline_fp16.om \
  --framework=5 \
  --soc_version=Ascend310B4 \
  --precision_mode=force_fp16 \
  --insert_op_conf=aipp_yuv420sp.cfg \
  --log=error
