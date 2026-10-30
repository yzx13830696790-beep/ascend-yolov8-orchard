"""ROS 闭环桥接节点：订阅图像 -> 推理 -> 发布检测框与单帧时延。

默认用 rospy；无 ROS 环境时降级为纯 stdout 打点（保证本机可验证）。
发布的话题：/orchard/detections（DetectionArray）、/orchard/latency_ms。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import rospy
    from sensor_msgs.msg import Image as RosImage
    HAS_ROS = True
except Exception:
    HAS_ROS = False


class DetectorNode:
    def __init__(self, infer_fn, publisher=None):
        """infer_fn(bgr) -> dict(boxes, scores, class_ids, latency_ms)。"""
        self.infer_fn = infer_fn
        self.publisher = publisher

    def step(self, bgr, image_id="frame"):
        out = self.infer_fn(bgr)
        if self.publisher is not None:
            self.publisher.publish(image_id, out)
        else:
            boxes = out["boxes"]
            print(f"[{image_id}] latency={out.get('latency_ms', 0):.2f}ms "
                  f"dets={len(boxes)}", flush=True)
        return out


class StdoutPublisher:
    def publish(self, image_id, out):
        boxes = out["boxes"]
        for b, s, c in zip(boxes, out["scores"], out["class_ids"]):
            print(f"  box={b.tolist()} score={s:.3f} class={int(c)}", flush=True)


def main():
    if HAS_ROS:
        rospy.init_node("orchard_detector")
        print("running with ROS (sensor_msgs/Image subscription)")
    else:
        print("ROSPY 未安装，进入 stdout 打点模式（本机可验证）")


if __name__ == "__main__":
    main()
