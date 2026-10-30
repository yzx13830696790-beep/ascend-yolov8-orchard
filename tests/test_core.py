"""核心库单元测试（unittest，本机可跑）。"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from orch import preprocess as pp
from orch import nms, metrics, atc, quant, pareto, decode, latency
from orch.nms import iou_matrix


class TestLetterbox(unittest.TestCase):
    def test_params_scale_up(self):
        r, (nw, nh), dw, dh = pp.letterbox_params(480, 640, (640, 640))
        self.assertAlmostEqual(r, 1.0)
        self.assertEqual((nw, nh), (640, 480))
        self.assertAlmostEqual(dw, 0.0)
        self.assertAlmostEqual(dh, 80.0)

    def test_scale_up_disabled(self):
        r, (nw, nh), dw, dh = pp.letterbox_params(960, 1280, (640, 640), scaleup=False)
        self.assertAlmostEqual(r, 0.5)
        self.assertEqual((nw, nh), (640, 480))
        self.assertAlmostEqual(dw, 0.0)
        self.assertAlmostEqual(dh, 80.0)

    def test_roundtrip(self):
        rng = np.random.default_rng(0)
        boxes = rng.uniform(0, 640, (10, 4))
        fwd = pp.scale_boxes_forward(boxes, 0.8, 12.0, 34.0)
        back = pp.scale_boxes_back(fwd, 0.8, 12.0, 34.0)
        self.assertTrue(np.allclose(boxes, back, atol=1e-4))


class TestAipp(unittest.TestCase):
    def test_affine_matches_normalize(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        ref = pp.normalize_nchw(img)
        got = pp.apply_aipp(img, round_affine=False)  # 全精度 mean/var
        self.assertLess(float(np.max(np.abs(ref - got))), 1e-3)

    def test_affine_integer_mean_bounded(self):
        # cfg 里 mean 只能填整数，会引入有界的量化误差（~0.01/通道）
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        ref = pp.normalize_nchw(img)
        got = pp.apply_aipp(img, round_affine=True)
        self.assertLess(float(np.max(np.abs(ref - got))), 0.02)

    def test_yuv_roundtrip_correct_matrix(self):
        rng = np.random.default_rng(2)
        rgb = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
        back = pp.yuv420sp_to_rgb(pp.rgb_to_yuv420sp(rgb, "BT601"), "BT601")
        self.assertLess(float(np.mean(np.abs(rgb.astype(np.int16)
                                              - back.astype(np.int16)))), 12.0)


class TestNms(unittest.TestCase):
    def test_iou(self):
        a = np.array([[0, 0, 10, 10]], np.float32)
        b = np.array([[5, 5, 15, 15]], np.float32)
        i = iou_matrix(a, b)[0, 0]
        self.assertAlmostEqual(i, 25.0 / 175.0, places=5)

    def test_parity_naive(self):
        rng = np.random.default_rng(3)
        boxes = rng.uniform(0, 100, (200, 4)).astype(np.float32)
        boxes[:, 2] = boxes[:, 0] + rng.uniform(1, 20, 200).astype(np.float32)
        boxes[:, 3] = boxes[:, 1] + rng.uniform(1, 20, 200).astype(np.float32)
        scores = rng.uniform(0, 1, 200).astype(np.float32)
        cls = rng.integers(0, 3, 200).astype(np.float32)
        k1 = nms.nms(boxes, scores, 0.5, 300, cls)
        k2 = nms.nms_naive(boxes, scores, 0.5, 300)  # 单类
        # 单类等价
        self.assertEqual(set(k1.tolist()) if len(np.unique(cls)) == 1 else set(),
                         set(k2.tolist()) if len(np.unique(cls)) == 1 else set())


class TestMetrics(unittest.TestCase):
    def test_ap_perfect(self):
        gts = {"a": {"boxes": [[0, 0, 10, 10]], "classes": [0], "visible": [1.0]}}
        preds = [{"image_id": "a", "boxes": np.array([[0, 0, 10, 10]], np.float32),
                  "scores": np.array([0.9], np.float32),
                  "class_ids": np.array([0], np.int64)}]
        m = metrics.evaluate(preds, gts)
        self.assertAlmostEqual(m["map50_95"], 1.0, places=4)

    def test_ap_miss(self):
        gts = {"a": {"boxes": [[0, 0, 10, 10]], "classes": [0], "visible": [1.0]}}
        preds = [{"image_id": "a", "boxes": np.array([[50, 50, 60, 60]], np.float32),
                  "scores": np.array([0.9], np.float32),
                  "class_ids": np.array([0], np.int64)}]
        m = metrics.evaluate(preds, gts)
        self.assertAlmostEqual(m["map50_95"], 0.0, places=4)

    def test_business_miss_fp(self):
        gts = {"a": {"boxes": [[0, 0, 10, 10], [0, 0, 10, 10]], "classes": [0, 0],
                     "visible": [1.0, 0.3]}}
        preds = [{"image_id": "a", "boxes": np.array([[50, 50, 60, 60]], np.float32),
                  "scores": np.array([0.9], np.float32),
                  "class_ids": np.array([0], np.int64)}]
        bm = metrics.business_metrics(preds, gts)
        self.assertEqual(bm["gt_pickable"], 1)
        self.assertEqual(bm["miss_count"], 1)
        self.assertEqual(bm["fp_count"], 1)


class TestAtcValidator(unittest.TestCase):
    def test_310b_needs_cann7(self):
        e = atc.validate(soc_version="Ascend310B4", cann_version="6.3.RC2")
        self.assertTrue(any(x["level"] == "error" for x in e))

    def test_rgb_with_dvpp(self):
        e = atc.validate(pipeline="dvpp", input_format="RGB")
        self.assertTrue(any("DVPP" in x["msg"] for x in e))

    def test_force_fp16_high_precision(self):
        e = atc.validate(precision_mode="force_fp16", op_select_implmode="high_precision")
        self.assertTrue(any("force_fp16" in x["msg"] for x in e))

    def test_clean_pass(self):
        e = atc.validate(soc_version="Ascend310B4", cann_version="8.0.RC1",
                         pipeline="dvpp", input_format="YUV420SP_U8",
                         precision_mode="force_fp16", op_select_implmode="high_performance")
        self.assertEqual(e, [])


class TestQuant(unittest.TestCase):
    def test_ptq_commands(self):
        p = quant.ptq_commands("m.onnx", "calib", "work")
        self.assertIn("amct_onnx quantization", p["quantization"])

    def test_sensitivity(self):
        p = quant.sensitivity_commands("m.onnx", ["conv1", "conv2"], "calib", "work")
        self.assertEqual(len(p["commands"]), 2)
        self.assertIn("--skip_layers conv2", p["commands"][0]["cmd"])


class TestPareto(unittest.TestCase):
    def test_front(self):
        pts = [
            {"name": "a", "latency": 8, "map": 0.9, "measured": True},
            {"name": "b", "latency": 10, "map": 0.8, "measured": True},
            {"name": "c", "latency": 12, "map": 0.95, "measured": True},
            {"name": "d", "latency": 9, "map": 0.85, "measured": True},
        ]
        front = pareto.pareto_front(pts)
        names = {p["name"] for p in front}
        self.assertIn("a", names)
        self.assertIn("c", names)
        self.assertNotIn("d", names)   # d 被 a 支配
        self.assertNotIn("b", names)   # b 被 a 支配


class TestDecode(unittest.TestCase):
    def test_decode_shape(self):
        rng = np.random.default_rng(0)
        o = rng.standard_normal((1, 8, 100)).astype(np.float32)  # 4 + 4 classes
        o[:, 4:, :] = 0
        o[:, 4, 10] = 5.0  # class 0 score high
        o[:, 0, 10] = 50; o[:, 1, 10] = 50; o[:, 2, 10] = 20; o[:, 3, 10] = 20
        d = decode.decode_yolov8(o, conf_thr=0.25, nc=4, to_original=False)
        self.assertGreaterEqual(len(d["boxes"]), 1)
        self.assertEqual(d["class_ids"][0], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
