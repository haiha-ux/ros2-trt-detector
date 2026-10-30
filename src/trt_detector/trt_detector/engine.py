"""GPU letterbox preprocessing, TensorRT runner and NMS decode for YOLO11 (adapted from a multi-stream pipeline repo).

preprocess_rgb  uint8 HxWx3 numpy frame -> letterboxed RGB float [0,1] 3xSxS on the GPU
TrtEngine       plain TensorRT 10 runner (torch tensors as I/O buffers)
decode_nms      raw YOLO output (B, 4+nc, A) -> [x1,y1,x2,y2,conf,cls] in source pixels
"""
import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms

PAD = 114 / 255  # Ultralytics letterbox grey


def letterbox_params(h, w, size):
    """Scale, (left, top) padding and resized (nh, nw) used to fit an h x w image into size x size."""
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    return r, (size - nw) // 2, (size - nh) // 2, nh, nw


def preprocess_rgb(frame, is_bgr, size, out):
    """frame: (H, W, C) uint8 numpy, C in {1, 3}. Writes the letterboxed RGB image into out (1,3,size,size).

    The uint8 frame is copied to the GPU first and resized there (bilinear, antialiased).
    Returns (r, left, top) for mapping boxes back to source pixels.
    """
    h, w, c = frame.shape
    r, left, top, nh, nw = letterbox_params(h, w, size)
    x = torch.from_numpy(frame).cuda(non_blocking=True).permute(2, 0, 1)[None]  # (1,C,H,W) uint8
    if c == 1:
        x = x.expand(-1, 3, -1, -1)
    elif is_bgr:
        x = x.flip(1)
    x = F.interpolate(x.float(), (nh, nw), mode="bilinear", antialias=True) / 255
    out.fill_(PAD)
    out[:, :, top:top + nh, left:left + nw] = x
    return r, left, top


def decode_nms(raw, metas, conf_thres=0.25, iou_thres=0.7, classes=(0,), max_det=300, pre_k=300):
    """raw: (B, 4+nc, A) fp32 (xywh in letterbox pixels + class scores). metas: [(r, left, top)] per image.

    GPU: best class score per anchor, top-k anchors per image, ONE device->host copy.
    CPU: confidence filter + class-aware NMS on at most pre_k boxes per image.
    """
    B = raw.shape[0]
    scores = raw[:, 4:].transpose(1, 2)[..., list(classes)]      # (B, A, nc_sel)
    conf, cls_i = scores.max(-1)                                  # (B, A)
    top_c, idx = conf.topk(min(pre_k, conf.shape[1]), dim=1)      # (B, K)
    boxes = torch.gather(raw[:, :4].transpose(1, 2), 1, idx[..., None].expand(-1, -1, 4))
    cls_top = torch.gather(cls_i, 1, idx)
    packed = torch.cat([boxes, top_c[..., None], cls_top[..., None].float()], -1).cpu()  # (B, K, 6)
    cls_ids = torch.tensor(list(classes), dtype=torch.float32)
    out = []
    for i in range(B):
        d = packed[i]
        d = d[d[:, 4] > conf_thres]
        if len(d) == 0:
            out.append(np.zeros((0, 6), np.float32))
            continue
        xyxy = torch.stack([d[:, 0] - d[:, 2] / 2, d[:, 1] - d[:, 3] / 2, d[:, 0] + d[:, 2] / 2, d[:, 1] + d[:, 3] / 2], 1)
        keep = batched_nms(xyxy, d[:, 4], d[:, 5].long(), iou_thres)[:max_det]
        r, left, top = metas[i]
        xyxy = xyxy[keep]
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - left) / r
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - top) / r
        out.append(torch.cat([xyxy, d[keep, 4:5], cls_ids[d[keep, 5].long()][:, None]], 1).numpy())
    return out


class TrtEngine:
    """TensorRT engine with a (dynamic) batch dimension, buffers preallocated for the max batch."""

    def __init__(self, plan_path):
        rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(plan_path, "rb") as f:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"could not deserialize {plan_path} (built with a different TensorRT version?)")
        self.ctx = self.engine.create_execution_context()
        self.inp, self.outp = self.engine.get_tensor_name(0), self.engine.get_tensor_name(1)
        self.max_batch = self.engine.get_tensor_profile_shape(self.inp, 0)[2][0]
        self.size = self.engine.get_tensor_shape(self.inp)[-1]
        out_shape = list(self.engine.get_tensor_shape(self.outp))
        dt = lambda n: torch.float16 if self.engine.get_tensor_dtype(n) == trt.float16 else torch.float32
        self.x = torch.zeros(self.max_batch, 3, self.size, self.size, device="cuda", dtype=dt(self.inp))
        out_shape[0] = self.max_batch
        self.y = torch.zeros(out_shape, device="cuda", dtype=dt(self.outp))

    def infer(self, n):
        """Run the first n images of self.x; returns a view of the first n outputs (fp32)."""
        self.ctx.set_input_shape(self.inp, (n, 3, self.size, self.size))
        self.ctx.set_tensor_address(self.inp, self.x.data_ptr())
        self.ctx.set_tensor_address(self.outp, self.y.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self.y[:n].float()
