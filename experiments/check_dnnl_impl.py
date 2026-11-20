"""교란 확인: 224와 448에서 oneDNN convolution이 어떤 구현을 선택하는지 DNNL_VERBOSE 로그로 비교."""

import collections
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
PY = ROOT / ".venv/Scripts/python.exe"

RUNNER = """
import ctypes, sys
from pathlib import Path
import numpy as np, onnxruntime as ort
ctypes.CDLL(str(Path('.venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll').resolve()))
so = ort.SessionOptions()
so.register_custom_ops_library(sys.argv[2])
s = ort.InferenceSession(sys.argv[1], so)
h, w = s.get_inputs()[0].shape[2], s.get_inputs()[0].shape[3]
s.run(None, {'input': np.zeros((1, 3, h, w), np.float32)})
"""

CONFIGS = [
    ("224", "models/mobilenetv2_cpp.onnx"),
    ("448", "models/_test_cpp_448.onnx"),
]
DLLS = [
    "cpp_backend/inverted_residual_op_dnnl_unfused.dll",
    "cpp_backend/inverted_residual_op_dnnl_fused.dll",
]


def impl_counts(model: str, dll: str) -> collections.Counter:
    env = dict(os.environ, DNNL_VERBOSE="1")
    proc = subprocess.run(
        [str(PY), "-c", RUNNER, model, dll],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    counts = collections.Counter()
    for line in (proc.stdout + proc.stderr).splitlines():
        parts = line.split(",")
        if len(parts) > 6 and parts[0] == "onednn_verbose" and parts[3] == "exec" and parts[5] == "convolution":
            counts[parts[6]] += 1
    if proc.returncode != 0:
        print("실행 실패:", proc.stderr[-500:], file=sys.stderr)
    return counts


def main() -> None:
    for res, model in CONFIGS:
        for dll in DLLS:
            print(f"== {res} | {Path(dll).name}")
            for impl, n in impl_counts(model, dll).most_common():
                print(f"   {n:4d} x {impl}")


if __name__ == "__main__":
    main()
