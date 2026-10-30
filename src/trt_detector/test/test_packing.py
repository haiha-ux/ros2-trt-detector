import numpy as np
import pytest

from trt_detector.coco import NAMES, class_ids
from trt_detector.packing import image_to_numpy, pack_detections


def test_image_to_numpy_bgr_is_zero_copy_and_respects_step():
    h, w, step = 4, 5, 20  # 15 bytes of pixels + 5 padding per row
    raw = bytearray(range(h * step))
    img, is_bgr = image_to_numpy(raw, h, w, step, "bgr8")
    assert img.shape == (h, w, 3) and is_bgr
    assert img[1, 0, 0] == raw[step] and img[1, 4, 2] == raw[step + 14]
    raw[0] = 200
    assert img[0, 0, 0] == 200  # view, not a copy


def test_image_to_numpy_mono_and_rgb():
    img, is_bgr = image_to_numpy(bytes(6), 2, 3, 3, "mono8")
    assert img.shape == (2, 3, 1) and not is_bgr
    img, is_bgr = image_to_numpy(bytes(18), 2, 3, 9, "rgb8")
    assert img.shape == (2, 3, 3) and not is_bgr


def test_image_to_numpy_rejects_bad_input():
    with pytest.raises(ValueError):
        image_to_numpy(bytes(100), 2, 3, 9, "bayer_rggb8")
    with pytest.raises(ValueError):
        image_to_numpy(bytes(10), 2, 3, 9, "rgb8")  # buffer too short


def test_pack_detections_center_size_class_and_threshold():
    dets = np.array([[10, 20, 50, 100, 0.9, 0], [0, 0, 8, 8, 0.1, 2]], np.float32)
    out = pack_detections(dets, NAMES, min_score=0.25)
    assert out == [(30.0, 60.0, 40.0, 80.0, pytest.approx(0.9), "person")]
    assert pack_detections(np.zeros((0, 6)), NAMES) == []


def test_class_ids():
    assert class_ids(["person", "car"]) == (0, 2)
    assert len(class_ids([])) == 80
    with pytest.raises(ValueError):
        class_ids(["unicorn"])
