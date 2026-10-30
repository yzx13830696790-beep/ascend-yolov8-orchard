"""生成正确的 AIPP 配置与 ATC 命令，并现场演示校验器能抓到任务书的三处硬伤。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orch import atc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    # 1) 正确配置
    aipp_cfg = atc.build_aipp_cfg(soc_version="Ascend310B4", pipeline="dvpp",
                                  matrix="BT601")
    atc_dir = os.path.join(ROOT, "deploy", "atc")
    os.makedirs(atc_dir, exist_ok=True)
    cfg_path = os.path.join(atc_dir, "aipp_yuv420sp.cfg")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(aipp_cfg)

    cmd = atc.build_atc_cmd("baseline.onnx", "baseline_fp16.om",
                            soc_version="Ascend310B4",
                            precision_mode="force_fp16",
                            aipp_cfg="aipp_yuv420sp.cfg")
    sh = "#!/bin/bash\nset -e\nsource /usr/local/Ascend/ascend-toolkit/set_env.sh\n\n" \
         + cmd + "\n"
    with open(os.path.join(atc_dir, "build_om.sh"), "w", encoding="utf-8") as f:
        f.write(sh)

    # 2) 校验：正确配置应零错误
    clean = atc.validate(soc_version="Ascend310B4", cann_version="8.0.RC1",
                         pipeline="dvpp", input_format="YUV420SP_U8",
                         precision_mode="force_fp16", op_select_implmode="high_performance",
                         matrix="BT601", train_matrix="BT601")
    # 3) 任务书里的错误配置，应被抓出
    wrong = atc.validate(soc_version="Ascend310B4", cann_version="6.3.RC2",
                         pipeline="dvpp", input_format="RGB",
                         precision_mode="force_fp16", op_select_implmode="high_precision",
                         matrix="BT709", train_matrix="BT601")

    result = {
        "correct_config_errors": clean,
        "taskbook_wrong_config_errors": [e["msg"] for e in wrong],
    }
    out = os.path.join(ROOT, "results", "atc_validation.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"aipp cfg -> {cfg_path}")
    print(f"build script -> {os.path.join(atc_dir, 'build_om.sh')}")
    print("\n正确配置校验：", "PASS（0 错误）" if not clean else clean)
    print("\n任务书错误配置被抓到的问题：")
    for e in wrong:
        print(f"  [{e['level']}] {e['msg']}")


if __name__ == "__main__":
    main()
