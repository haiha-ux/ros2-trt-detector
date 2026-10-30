"""results/bench.json -> results/bench.png (stage times and latency per input size)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
res = json.loads((ROOT / "results/bench.json").read_text())
stages = ["convert_ms", "preprocess_ms", "infer_ms", "nms_ms", "publish_ms"]
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
bottom = [0] * len(res)
for s in stages:
    v = [r["stages_ms"][s]["median"] for r in res]
    a.bar([str(r["input_size"]) for r in res], v, bottom=bottom, label=s[:-3])
    bottom = [x + y for x, y in zip(bottom, v)]
a.set(title="Node-internal stage time (median)", xlabel="engine input size (px)", ylabel="ms")
a.legend(loc="upper left"); a.set_ylim(0, max(bottom) * 1.35)
x = [str(r["input_size"]) for r in res]
med = [r["stamp_to_receive_ms"]["median"] for r in res]
p95 = [r["stamp_to_receive_ms"]["p95"] for r in res]
b.bar([i - 0.2 for i in range(len(x))], med, 0.4, label="median")
b.bar([i + 0.2 for i in range(len(x))], p95, 0.4, label="p95")
b.set(xticks=range(len(x)), xticklabels=x, title="Header stamp -> detection received", xlabel="engine input size (px)", ylabel="ms")
b.legend(loc="upper left"); b.set_ylim(0, max(p95) * 1.25)
for r, i in zip(res, range(len(x))):
    b.text(i, max(p95) * 1.08, f"{r['detection_rate_hz']:.1f} Hz", ha="center", fontsize=8)
fig.suptitle("ros2-trt-detector, 30 Hz input (shared GPU, relative numbers)")
fig.tight_layout()
fig.savefig(ROOT / "results/bench.png", dpi=110)
