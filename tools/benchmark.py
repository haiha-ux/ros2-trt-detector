"""End-to-end benchmark of the detector node at 30 Hz input, one run per engine size.

Run from the repo root with the workspace sourced:
    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    nice -n 19 python3 tools/benchmark.py --video <clip.mp4> --sizes 640 1280

Per size it starts video_publisher + detector_node as separate processes, then measures in this process:
  stamp_to_receive  image header stamp -> detection message received here (camera-to-consumer, includes DDS)
  detector stage times (convert/preprocess/infer/nms/publish) from the node's own timing CSV
  detection rate, publisher rate, CPU use of both processes (% of one core)
"""
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent.parent
os.environ["ROS_DOMAIN_ID"] = "77"  # isolate from other ROS traffic on the machine
# 1280x720 images are 2.7 MB: Fast DDS default transports drop most of them, a larger shared-memory segment does not
os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = str(ROOT / "config/fastdds_shm.xml")

import rclpy  # noqa: E402  (after the env is set)
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from vision_msgs.msg import Detection2DArray  # noqa: E402

STAGES = ["convert_ms", "preprocess_ms", "infer_ms", "nms_ms", "publish_ms", "total_ms", "stamp_to_publish_ms"]


def stats(x):
    x = np.asarray(x)
    return {"median": float(np.median(x)), "p95": float(np.percentile(x, 95)), "mean": float(x.mean())}


def gpu_util():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.split()[0]
        return float(out)
    except Exception:
        return None


def stop(proc):
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(30)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_one(size, a, tmp):
    pub_stats, timing = tmp / f"pub_{size}.json", tmp / f"timing_{size}.csv"
    ros = lambda *kv: [x for k, v in kv for x in ("-p", f"{k}:={v}")]
    mod = lambda m, *kv: [sys.executable, "-m", f"trt_detector.{m}", "--ros-args", *ros(*kv)]
    pub = subprocess.Popen(mod("video_publisher", ("video_path", a.video), ("rate_hz", a.rate), ("width", a.width),
                               ("height", a.height), ("stats_json", pub_stats)))
    det = subprocess.Popen(mod("detector_node", ("engine_path", ROOT / f"engines/yolo11n_fp16_{size}.plan"),
                               ("timing_csv", timing), ("publish_annotated_image", False)))
    node = rclpy.create_node(f"bench_{size}")
    lat, recv = [], []

    def cb(msg):
        now = node.get_clock().now().nanoseconds
        lat.append((now - msg.header.stamp.sec * 10**9 - msg.header.stamp.nanosec) / 1e6)
        recv.append(now)
    node.create_subscription(Detection2DArray, "/detections", cb, qos_profile_sensor_data)
    try:
        t_end = time.monotonic() + 90  # engine load + first frame
        while not lat and time.monotonic() < t_end:
            rclpy.spin_once(node, timeout_sec=0.2)
        if not lat:
            raise RuntimeError("no detections within 90 s")
        t_end = time.monotonic() + a.warmup
        while time.monotonic() < t_end:
            rclpy.spin_once(node, timeout_sec=0.2)
        lat.clear(), recv.clear()
        procs = [psutil.Process(det.pid), psutil.Process(pub.pid)]
        [p.cpu_percent(None) for p in procs]
        util_before = gpu_util()
        t0, t_ns = time.monotonic(), node.get_clock().now().nanoseconds
        while time.monotonic() - t0 < a.duration:
            rclpy.spin_once(node, timeout_sec=0.2)
        dt = time.monotonic() - t0
        cpu_det, cpu_pub = (p.cpu_percent(None) for p in procs)
    finally:
        node.destroy_node()
        stop(det), stop(pub)
    rows = [r for r in csv.DictReader(open(timing)) if int(r["stamp_ns"]) >= t_ns]
    shutil_copy(timing, ROOT / f"results/frames_{size}.csv", t_ns)
    ps = json.load(open(pub_stats))
    return {
        "input_size": size, "source": f"{a.width}x{a.height}", "input_rate_hz": a.rate,
        "publisher_rate_hz": ps["rate_hz"], "detection_rate_hz": len(lat) / dt, "frames": len(lat),
        "stamp_to_receive_ms": stats(lat),
        "stages_ms": {s: stats([float(r[s]) for r in rows]) for s in STAGES},
        "mean_detections_per_frame": float(np.mean([int(r["n_det"]) for r in rows])),
        "cpu_detector_pct": cpu_det, "cpu_publisher_pct": cpu_pub, "gpu_util_before_pct": util_before,
    }


def shutil_copy(src, dst, t_ns):
    """Keep only the measurement window in the committed per-frame CSV."""
    with open(src) as f, open(dst, "w", newline="") as g:
        r = csv.reader(f)
        w = csv.writer(g)
        w.writerow(next(r))
        w.writerows(row for row in r if int(row[0]) >= t_ns)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[640, 1280])
    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--warmup", type=float, default=8)
    a = ap.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "trt_bench"
    tmp.mkdir(exist_ok=True)
    rclpy.init()
    res = []
    for s in a.sizes:
        print(f"== input {s}", flush=True)
        res.append(run_one(s, a, tmp))
        print(json.dumps(res[-1]["stamp_to_receive_ms"]), flush=True)
    rclpy.shutdown()
    (ROOT / "results/bench.json").write_text(json.dumps(res, indent=2))
    with open(ROOT / "results/bench.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["input_size", "detection_rate_hz", "stamp_to_receive_median_ms", "stamp_to_receive_p95_ms",
                    *[f"{s}_{k}" for s in STAGES for k in ("median", "p95")], "cpu_detector_pct", "cpu_publisher_pct"])
        for r in res:
            w.writerow([r["input_size"], round(r["detection_rate_hz"], 2), *(round(r["stamp_to_receive_ms"][k], 2) for k in ("median", "p95")),
                        *(round(r["stages_ms"][s][k], 2) for s in STAGES for k in ("median", "p95")),
                        round(r["cpu_detector_pct"], 1), round(r["cpu_publisher_pct"], 1)])
