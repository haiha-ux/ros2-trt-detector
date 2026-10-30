"""Test source: replay a video file as sensor_msgs/Image at a fixed rate (decode in a background thread)."""
import json
import queue
import threading
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image


class VideoPublisher(Node):
    def __init__(self):
        super().__init__("video_publisher")
        p = self.declare_parameter
        self.path = p("video_path", "").value
        topic = p("image_topic", "/image_raw").value
        rate = p("rate_hz", 30.0).value
        self.size = (p("width", 1280).value, p("height", 720).value)
        self.stats_json = p("stats_json", "").value
        if not self.path or not cv2.VideoCapture(self.path).isOpened():
            raise RuntimeError(f"cannot open video_path={self.path!r}")
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub = self.create_publisher(Image, topic, qos)
        self.q = queue.Queue(maxsize=8)
        self.stop = threading.Event()
        threading.Thread(target=self._reader, daemon=True).start()
        self.sent = self.missed = 0
        self.t_start = time.monotonic()
        self.create_timer(1.0 / rate, self.tick)

    def _reader(self):
        cap = cv2.VideoCapture(self.path)
        while not self.stop.is_set():
            ok, frame = cap.read()
            if not ok:  # loop the clip
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            self.q.put(cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA))

    def tick(self):
        try:
            frame = self.q.get_nowait()
        except queue.Empty:
            self.missed += 1
            return
        msg = Image(height=frame.shape[0], width=frame.shape[1], encoding="bgr8", step=frame.shape[1] * 3)
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        msg.data = frame.tobytes()
        self.pub.publish(msg)
        self.sent += 1

    def write_stats(self):
        if self.stats_json:
            dt = time.monotonic() - self.t_start
            with open(self.stats_json, "w") as f:
                json.dump({"published": self.sent, "missed_ticks": self.missed, "rate_hz": self.sent / dt}, f)


def main():
    rclpy.init()
    node = VideoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop.set()
        node.write_stats()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
