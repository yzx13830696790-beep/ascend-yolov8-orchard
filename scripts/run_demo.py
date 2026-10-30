"""一键端到端演示：数据 -> 清单 -> 评测 -> 实验 -> 帕累托 -> ATC 校验。

用法：python scripts/run_demo.py
（数据已存在则跳过生成；用 --force 重新生成）
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def run(name, *args):
    print(f"\n===== {name} =====")
    r = subprocess.run([PY, *args], cwd=ROOT)
    if r.returncode != 0:
        print(f"[fail] {name}")
        sys.exit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    imgs = os.path.join(ROOT, "data", "images")
    if args.force or not os.path.isdir(imgs) or not os.listdir(imgs):
        run("1/6 生成合成数据", "scripts/make_demo_data.py")
    run("2/6 建清单/划分/校准集", "scripts/prepare_data.py")
    run("3/6 评测", "scripts/run_eval.py")
    run("4/6 实验：AIPP 色域/归一化错配", "scripts/exp_color_matrix.py")
    run("5/6 实验：NMS 基准", "scripts/exp_nms_bench.py")
    run("6/6 帕累托 + ATC 校验", "scripts/make_pareto.py")
    run("6/6 ATC 配置校验", "scripts/gen_atc.py")

    print("\n全部完成。产物：")
    for p in ["results/metrics_val.json", "results/report_val.md",
              "results/ledger.csv", "results/exp_aipp_color_mismatch.json",
              "results/exp_nms_bench.json", "results/pareto_points.csv",
              "results/figures/pareto.png", "results/atc_validation.json",
              "deploy/atc/aipp_yuv420sp.cfg", "deploy/atc/build_om.sh"]:
        print("  -", p)


if __name__ == "__main__":
    main()
