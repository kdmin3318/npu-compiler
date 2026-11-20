"""VTune 프로파일링 대상: unfused+oneDNN (타일링 없음). 반복 추론만 수행."""

import ctypes
from pathlib import Path

import numpy as np
import onnxruntime as ort

_ORT_DLL = Path(__file__).parent.parent / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
ctypes.CDLL(str(_ORT_DLL))

so = ort.SessionOptions()
so.register_custom_ops_library("cpp_backend/inverted_residual_op_dnnl_unfused.dll")
sess = ort.InferenceSession("models/mobilenetv2_cpp.onnx", so)

sample = np.random.default_rng(0).standard_normal((1, 3, 224, 224)).astype(np.float32)

for _ in range(5):
    sess.run(None, {"input": sample})  # warmup

for _ in range(200):
    sess.run(None, {"input": sample})
