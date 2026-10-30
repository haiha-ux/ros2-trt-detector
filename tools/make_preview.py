"""Record docs/preview.gif from the running pipeline: raw image + /detections matched by header stamp,
head regions (top 18% of every person box) blurred, boxes drawn, downscaled to 400 px wide, 5 fps.

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    python3 tools/make_preview.py --video <clip.mp4>
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parent.parent
os.environ["ROS_DOMAIN_ID"] = "78"
os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = str(ROOT / "config/fastdds_shm.xml")

import rclpy  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402
from vision_msgs.msg import Detection2DArray  # noqa: E402

from trt_detector.packing import image_to_numpy  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--seconds", type=float, default=6)
ap.add_argument("--out", default=str(ROOT / "docs/preview.gif"))
a = ap.parse_args()

mod = lambda m, *kv: [sys.executable, "-m", f"trt_detector.{m}", "--ros-args", *[x for k, v in kv for x in ("-p", f"{k}:={v}")]]
procs = [subprocess.Popen(mod("video_publisher", ("video_path", a.video))),
         subprocess.Popen(mod("detector_node", ("engine_path", ROOT / "engines/yolo11n_fp16_640.plan"), ("publish_annotated_image", True)))]
rclpy.init()
node = rclpy.create_node("preview")
imgs, dets, annotated = {}, {}, [0]
key = lambda m: (m.header.stamp.sec, m.header.stamp.nanosec)


def on_img(m):
    arr, _ = image_to_numpy(m.data, m.height, m.width, m.step, m.encoding)
    imgs[key(m)] = arr.copy()
    for k in list(imgs)[:-20]:
        del imgs[k]


node.create_subscription(Image, "/image_raw", on_img, qos_profile_sensor_data)
node.create_subscription(Detection2DArray, "/detections", lambda m: dets.__setitem__(key(m), m), qos_profile_sensor_data)
node.create_subscription(Image, "/detections/image", lambda m: annotated.__setitem__(0, annotated[0] + 1), qos_profile_sensor_data)
frames, done, t0 = [], set(), None
try:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if dets and t0 is None:
            t0 = time.monotonic() + 3  # let the stream settle
        if t0 and time.monotonic() > t0 + a.seconds:
            break
        if not t0 or time.monotonic() < t0:
            continue
        for k in sorted(dets):
            if k in imgs and k not in done:
                done.add(k)
                if len(done) % 6:  # 30 fps -> 5 fps
                    continue
                img = imgs[k].copy()
                for d in dets[k].detections:
                    cx, cy = d.bbox.center.position.x, d.bbox.center.position.y
                    w, h = d.bbox.size_x, d.bbox.size_y
                    x1, y1, x2, y2 = (max(0, int(v)) for v in (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
                    hy = y1 + max(4, int(0.18 * (y2 - y1)))
                    if hy > y1 and x2 > x1:
                        img[y1:hy, x1:x2] = cv2.GaussianBlur(img[y1:hy, x1:x2], (0, 0), 12)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                frames.append(PILImage.fromarray(cv2.cvtColor(cv2.resize(img, (400, 225), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)).quantize(64))
finally:
    node.destroy_node()
    rclpy.shutdown()
    for p in procs:
        p.send_signal(signal.SIGINT)
    for p in procs:
        p.wait(30)
assert frames, "no frames captured"
frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=200, loop=0, optimize=True)
print(f"{len(frames)} frames -> {a.out}; annotated images seen: {annotated[0]}")
