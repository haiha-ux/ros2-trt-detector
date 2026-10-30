"""ROS 2 node: sensor_msgs/Image -> TensorRT YOLO -> vision_msgs/Detection2DArray (+ optional annotated image)."""
import csv
import time

import cv2
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from trt_detector.coco import NAMES, class_ids
from trt_detector.engine import TrtEngine, decode_nms, preprocess_rgb
from trt_detector.packing import image_to_numpy, pack_detections

STAGES = ("convert", "preprocess", "infer", "nms", "publish")


class DetectorNode(Node):
    def __init__(self):
        super().__init__("trt_detector")
        p = self.declare_parameter
        engine_path = p("engine_path", "engines/yolo11n_fp16_640.plan").value
        self.image_topic = p("image_topic", "/image_raw").value
        self.score_thres = p("score_threshold", 0.25).value
        self.iou_thres = p("nms_iou_threshold", 0.7).value
        self.publish_image = p("publish_annotated_image", False).value
        self.timing_csv = p("timing_csv", "").value  # per-frame stage times, written on shutdown
        qos_depth = p("qos_depth", 1).value
        self.classes = class_ids(list(p("classes", ["person"]).value))
        self.max_det = p("max_detections", 100).value
        frac = p("gpu_memory_fraction", 0.3).value

        torch.cuda.set_per_process_memory_fraction(frac)
        self.engine = TrtEngine(engine_path)
        self.pre = torch.zeros(1, 3, self.engine.size, self.engine.size, device="cuda")
        self._warmup()

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=qos_depth)
        self.sub = self.create_subscription(Image, self.image_topic, self.on_image, qos)
        self.pub = self.create_publisher(Detection2DArray, "/detections", 10)
        self.pub_img = self.create_publisher(Image, "/detections/image", qos) if self.publish_image else None
        self.rows = []
        self.get_logger().info(
            f"engine {engine_path} ({self.engine.size}px) on {self.image_topic}, classes={[NAMES[i] for i in self.classes]}")

    def _warmup(self):
        dummy = np.zeros((480, 640, 3), np.uint8)
        for _ in range(3):
            self._detect(dummy, False)
        torch.cuda.synchronize()

    def _detect(self, frame, is_bgr, times=None):
        """frame -> (N,6) detections in source pixels. Fills times[stage] (ms) when given."""
        t = time.perf_counter()
        meta = preprocess_rgb(frame, is_bgr, self.engine.size, self.pre)
        self.engine.x[:1].copy_(self.pre)  # fp32 -> engine input dtype
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        raw = self.engine.infer(1)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        dets = decode_nms(raw, [meta], self.score_thres, self.iou_thres, self.classes, self.max_det)[0]
        t3 = time.perf_counter()
        if times is not None:
            times.update(preprocess=(t1 - t) * 1e3, infer=(t2 - t1) * 1e3, nms=(t3 - t2) * 1e3)
        return dets

    def on_image(self, msg):
        t0 = time.perf_counter()
        try:
            frame, is_bgr = image_to_numpy(msg.data, msg.height, msg.width, msg.step, msg.encoding)
        except ValueError as e:
            self.get_logger().warning(str(e), throttle_duration_sec=5.0)
            return
        times = {"convert": (time.perf_counter() - t0) * 1e3}
        dets = self._detect(frame, is_bgr, times)

        t = time.perf_counter()
        out = Detection2DArray(header=msg.header)  # stamp + frame_id copied from the input image
        for cx, cy, w, h, score, name in pack_detections(dets, NAMES):
            d = Detection2D(header=msg.header)
            d.bbox.center.position.x, d.bbox.center.position.y = cx, cy
            d.bbox.size_x, d.bbox.size_y = w, h
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id, hyp.hypothesis.score = name, score
            d.results.append(hyp)
            out.detections.append(d)
        self.pub.publish(out)
        if self.pub_img is not None and self.pub_img.get_subscription_count() > 0:
            self.pub_img.publish(self._annotate(msg, frame, is_bgr, dets))
        times["publish"] = (time.perf_counter() - t) * 1e3

        if self.timing_csv:
            now = self.get_clock().now().nanoseconds
            stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            total = (time.perf_counter() - t0) * 1e3
            self.rows.append([stamp, *(round(times[s], 3) for s in STAGES), round(total, 3), round((now - stamp) / 1e6, 3), len(out.detections)])

    @staticmethod
    def _annotate(msg, frame, is_bgr, dets):
        img = np.ascontiguousarray(frame if is_bgr or frame.shape[2] == 1 else frame[..., ::-1])
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2, score, cls in dets:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f"{NAMES[int(cls)]} {score:.2f}", (int(x1), max(12, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        out = Image(header=msg.header, height=img.shape[0], width=img.shape[1], encoding="bgr8", step=img.shape[1] * 3)
        out.data = img.tobytes()
        return out

    def write_timing(self):
        if self.timing_csv and self.rows:
            with open(self.timing_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["stamp_ns", *(f"{s}_ms" for s in STAGES), "total_ms", "stamp_to_publish_ms", "n_det"])
                w.writerows(self.rows)


def main():
    rclpy.init()
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.write_timing()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
