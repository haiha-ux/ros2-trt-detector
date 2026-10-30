"""YOLO11n .pt -> ONNX -> TensorRT FP16 engine at a fixed input size (adapted from a multi-stream pipeline repo).

    python tools/build_engine.py --imgsz 640     # -> engines/yolo11n_fp16_640.plan
    python tools/build_engine.py --imgsz 1280

Engines are tied to the GPU + TensorRT version they were built with, so build on the target device.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def export_onnx(pt, imgsz):
    from ultralytics import YOLO
    return Path(YOLO(pt).export(format="onnx", imgsz=imgsz, opset=17, dynamic=True, simplify=False))


def build(onnx, out, max_batch, imgsz):
    import tensorrt as trt
    log = trt.Logger(trt.Logger.WARNING)
    b = trt.Builder(log)
    net = b.create_network()
    if not trt.OnnxParser(net, log).parse(onnx.read_bytes()):
        raise SystemExit("ONNX parse failed")
    cfg = b.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    prof = b.create_optimization_profile()
    shape = lambda n: (n, 3, imgsz, imgsz)
    prof.set_shape(net.get_input(0).name, shape(1), shape(1), shape(max_batch))
    cfg.add_optimization_profile(prof)
    plan = b.build_serialized_network(net, cfg)
    if not plan:
        raise SystemExit("engine build failed")
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(bytes(plan))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "weights/yolo11n.pt"), help="Ultralytics .pt (downloaded if missing)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-batch", type=int, default=1)
    ap.add_argument("--onnx-only", action="store_true")
    a = ap.parse_args()
    onnx = ROOT / f"weights/yolo11n_{a.imgsz}.onnx"
    if a.onnx_only:  # separate process: Ultralytics' CPU ONNX export hides the GPU from the rest of that process
        export_onnx(a.weights, a.imgsz).rename(onnx)
        raise SystemExit
    subprocess.run([sys.executable, __file__, "--onnx-only", "--weights", a.weights, "--imgsz", str(a.imgsz)], check=True)
    out = ROOT / f"engines/yolo11n_fp16_{a.imgsz}.plan"
    build(onnx, out, a.max_batch, a.imgsz)
    print("engine:", out)
