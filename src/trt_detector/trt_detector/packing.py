"""Pure-numpy helpers between ROS messages and the detector (no rclpy / torch import, unit-testable)."""
import numpy as np

_CHANNELS = {"rgb8": 3, "bgr8": 3, "mono8": 1}


def image_to_numpy(data, height, width, step, encoding):
    """sensor_msgs/Image fields -> (H, W, C) uint8 view over the message buffer (no copy).

    Honors `step` (row stride, may include padding). Returns (array, is_bgr).
    Raises ValueError for unsupported encodings or a buffer that is too small.
    """
    if encoding not in _CHANNELS:
        raise ValueError(f"unsupported encoding {encoding!r}; use one of {sorted(_CHANNELS)}")
    c = _CHANNELS[encoding]
    if step < width * c or len(data) < height * step:
        raise ValueError("image buffer smaller than height*step")
    buf = np.frombuffer(data, dtype=np.uint8, count=height * step).reshape(height, step)
    return buf[:, : width * c].reshape(height, width, c), encoding == "bgr8"


def pack_detections(dets, names, min_score=0.0):
    """(N,6) [x1,y1,x2,y2,score,cls] in source pixels -> list of (cx, cy, w, h, score, class_name).

    Detections below min_score are dropped; boxes are returned in the Detection2D convention
    (centre + size, pixels).
    """
    out = []
    for x1, y1, x2, y2, score, cls in np.asarray(dets, dtype=np.float64).reshape(-1, 6):
        if score < min_score:
            continue
        out.append(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, float(score), names[int(cls)]))
    return out
