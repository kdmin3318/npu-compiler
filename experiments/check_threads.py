"""교란 3 확인: 224와 448에서 스레드 수에 따른 속도 향상이 같은지 본다.
OMP_NUM_THREADS를 1과 기본값(미설정)으로 각각 돌려서 비교.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
PY = ROOT / ".venv/Scripts/python.exe"

RUNNER = """
import ctypes, time, sys
from pathlib import Path
import numpy as np, onnxruntime as ort
ctypes.CDLL(str(Path('.venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll').resolve()))
so = ort.SessionOptions()
so.register_custom_ops_library(sys.argv[2])
s = ort.InferenceSession(sys.argv[1], so)
h, w = s.get_inputs()[0].shape[2], s.get_inputs()[0].shape[3]
x = np.random.default_rng(0).standard_normal((1, 3, h, w)).astype(np.float32)
for _ in range(3):
    s.run(None, {'input': x})
times = []
for _ in range(15):
    t = time.perf_counter(); s.run(None, {'input': x}); times.append(time.perf_counter() - t)
print(f'{sorted(times)[len(times)//2]*1000:.2f}')
"""

DLL = "cpp_backend/inverted_residual_op_dnnl_unfused.dll"
CONFIGS = [("224", "models/mobilenetv2_cpp.onnx"), ("448", "models/_test_cpp_448.onnx")]


def median_ms(model: str, threads: str | None) -> float:
    env = dict(os.environ)
    if threads is None:
        env.pop("OMP_NUM_THREADS", None)
    else:
        env["OMP_NUM_THREADS"] = threads
    proc = subprocess.run([str(PY), "-c", RUNNER, model, DLL], cwd=ROOT, env=env,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr[-500:], file=sys.stderr)
        raise SystemExit(1)
    return float(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    for res, model in CONFIGS:
        t1 = median_ms(model, "1")
        tdef = median_ms(model, None)
        print(f"{res}: 1스레드 {t1:.1f}ms, 기본 {tdef:.1f}ms, 향상 배율 {t1 / tdef:.2f}x")


if __name__ == "__main__":
    main()
