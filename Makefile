# orchard310b 项目命令入口。本机用默认 Python（3.8 带 numpy/cv2）。
PY ?= python

.PHONY: test demo data eval exps pareto atc serve clean

serve:                     ## 启动实验平台（http://127.0.0.1:8765）
	$(PY) console/server.py --port 8765

test:                      ## 跑核心库单元测试
	$(PY) -m unittest tests.test_core -v

demo:                      ## 一键端到端演示
	$(PY) scripts/run_demo.py

data:                      ## 重新生成合成数据
	$(PY) scripts/make_demo_data.py && $(PY) scripts/prepare_data.py

eval:                      ## 评测 val 集
	$(PY) scripts/run_eval.py

exps:                      ## 两个实验（AIPP 错配 / NMS 基准）
	$(PY) scripts/exp_color_matrix.py && $(PY) scripts/exp_nms_bench.py

pareto:                    ## 画帕累托图
	$(PY) scripts/make_pareto.py

atc:                       ## 生成 AIPP 配置 + ATC 命令 + 校验
	$(PY) scripts/gen_atc.py

clean:
	rm -rf data/images data/labels data/meta results/*.json results/*.csv results/report_val.md results/figures/*.png
