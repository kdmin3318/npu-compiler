"""캐시 regime 실험: 입력 224 vs 448에서 fused(예산별) / unfused 시간 비율 비교.

사전 기준(실험 전 고정):
- 주 지표: R(size) = fused_L3 / unfused, 개선률 = (R224 - R448) / R224
- 지지: 개선률 >= 20%, 3회 반복 모두 같은 방향
- 약한 지지: 10~20%, 또는 3회 중 2회
- 기각: < 10% 또는 방향 반대
- 통제: fused_untiled / unfused 비율도 같은 방식으로 본다 (타일링 효과와 구조 차이 분리)
"""

import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
PY = ROOT / ".venv/Scripts/python.exe"

RUNNER = """
import ctypes, time, sys, statistics
from pathlib import Path
import numpy as np, onnxruntime as ort
ctypes.CDLL(str(Path('.venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll').resolve()))
so = ort.SessionOptions()
if sys.argv[2] != 'none':
    so.register_custom_ops_library(sys.argv[2])
s = ort.InferenceSession(sys.argv[1], so)
h, w = s.get_inputs()[0].shape[2], s.get_inputs()[0].shape[3]
x = np.random.default_rng(0).standard_normal((1, 3, h, w)).astype(np.float32)
for _ in range(3):
    s.run(None, {'input': x})
times = []
for _ in range(15):
    t = time.perf_counter(); s.run(None, {'input': x}); times.append(time.perf_counter() - t)
print(f'{statistics.median(times)*1000:.3f}')
"""

MODELS = {"224": "models/mobilenetv2_cpp.onnx", "448": "models/_test_cpp_448.onnx"}
DLL_UNFUSED = "cpp_backend/inverted_residual_op_dnnl_unfused.dll"
DLL_FUSED = "cpp_backend/inverted_residual_op_dnnl_fused.dll"
BUDGETS = {"L2_2.5MB": 2621440, "L3_12MB": 12582912, "untiled": 1 << 30}
THREADS = {"default": None, "1thread": "1"}
REPEATS = 3


def run_once(model: str, dll: str, budget: int | None, threads: str | None) -> float:
    env = dict(os.environ)
    env.pop("OMP_NUM_THREADS", None)
    if threads is not None:
        env["OMP_NUM_THREADS"] = threads
    if budget is not None:
        env["INVRES_BUDGET_BYTES"] = str(budget)
    proc = subprocess.run([str(PY), "-c", RUNNER, model, dll], cwd=ROOT, env=env,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.stderr[-800:])
    return float(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    results = {}  # (thread, size, label) -> list of medians
    for thr_name, thr in THREADS.items():
        for size, model in MODELS.items():
            for _ in range(REPEATS):
                unf = run_once(model, DLL_UNFUSED, None, thr)
                results.setdefault((thr_name, size, "unfused"), []).append(unf)
                for label, budget in BUDGETS.items():
                    t = run_once(model, DLL_FUSED, budget, thr)
                    results.setdefault((thr_name, size, label), []).append(t)
            print(f"done {thr_name} {size}", flush=True)

    for thr_name in THREADS:
        print(f"\n== 스레드: {thr_name}")
        for label in BUDGETS:
            ratios = {}
            for size in MODELS:
                u = results[(thr_name, size, "unfused")]
                f = results[(thr_name, size, label)]
                ratios[size] = [fi / ui for fi, ui in zip(f, u)]
            r224 = statistics.median(ratios["224"])
            r448 = statistics.median(ratios["448"])
            improv = (r224 - r448) / r224 * 100
            per_rep = [(r224_i - r448_i) / r224_i * 100
                       for r224_i, r448_i in zip(ratios["224"], ratios["448"])]
            print(f"{label:>10}: R224={r224:.2f} R448={r448:.2f} 개선률={improv:+.1f}% "
                  f"(반복별 {', '.join(f'{p:+.1f}' for p in per_rep)})")


if __name__ == "__main__":
    main()
