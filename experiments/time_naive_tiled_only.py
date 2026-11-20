"""타일 크기 스윕용: fused+naive(타일링) DLL 하나만 로드해서 시간만 잰다."""

import ctypes
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

_ORT_DLL = Path(__file__).parent.parent / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
ctypes.CDLL(str(_ORT_DLL))

so = ort.SessionOptions()
so.register_custom_ops_library("cpp_backend/inverted_residual_op_naive_tiled.dll")
sess = ort.InferenceSession("models/mobilenetv2_cpp.onnx", so)

sample = np.random.default_rng(0).standard_normal((1, 3, 224, 224)).astype(np.float32)

for _ in range(5):
    sess.run(None, {"input": sample})

times = []
for _ in range(50):
    start = time.perf_counter()
    sess.run(None, {"input": sample})
    times.append(time.perf_counter() - start)

print(f"{np.mean(times)*1000:.3f} ms (+/- {np.std(times)*1000:.3f})")
