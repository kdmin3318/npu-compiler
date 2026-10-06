"""1차 성능 측정: 원본(unfused, ORT 기본 MLAS 실행) vs fused(우리 C++
커스텀 op) 전체 추론 시간(wall-clock) 비교.

주의: 이건 "공정한 비교"가 아닙니다 - MLAS(최적화됨)와 우리 naive
C++ 커널(최적화 안 됨)을 비교하는 거라, fusion 자체의 순수 효과와
구현 품질 차이가 섞여 있습니다. "지금 당장 실사용 대비 어디쯤
있는지" 보여주는 참고용 1차 측정치입니다.
"""

import ctypes
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

_ORT_DLL = Path(__file__).parent.parent / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
ctypes.CDLL(str(_ORT_DLL))

N_WARMUP = 5
N_RUNS = 50


def _time_session(sess: ort.InferenceSession, sample_input: np.ndarray) -> list[float]:
    for _ in range(N_WARMUP):
        sess.run(None, {"input": sample_input})

    times = []
    for _ in range(N_RUNS):
        start = time.perf_counter()
        sess.run(None, {"input": sample_input})
        times.append(time.perf_counter() - start)
    return times


def main() -> None:
    sample_input = np.random.default_rng(0).standard_normal((1, 3, 224, 224)).astype(np.float32)

    so_orig = ort.SessionOptions()
    orig_sess = ort.InferenceSession("models/mobilenetv2.onnx", so_orig)
    orig_times = _time_session(orig_sess, sample_input)

    so_cpp = ort.SessionOptions()
    so_cpp.register_custom_ops_library("cpp_backend/inverted_residual_op.dll")
    cpp_sess = ort.InferenceSession("models/mobilenetv2_cpp.onnx", so_cpp)
    cpp_times = _time_session(cpp_sess, sample_input)

    orig_mean, orig_std = np.mean(orig_times) * 1000, np.std(orig_times) * 1000
    cpp_mean, cpp_std = np.mean(cpp_times) * 1000, np.std(cpp_times) * 1000

    print(f"원본(unfused, MLAS): {orig_mean:.3f} ms (±{orig_std:.3f})")
    print(f"fused(우리 C++, naive): {cpp_mean:.3f} ms (±{cpp_std:.3f})")
    print(f"배율: {cpp_mean / orig_mean:.2f}x")

    with open("docs/backend_correctness_verification.md", "a", encoding="utf-8") as f:
        f.write("\n## 1차 성능 측정 (wall-clock, 참고용)\n\n")
        f.write("MLAS(원본, 최적화됨) vs 우리 naive C++ 커널(fused) 비교. ")
        f.write("공정한 비교 아님(구현 품질 차이 포함) - 참고용 1차 수치.\n\n")
        f.write(f"- 원본(unfused, MLAS): {orig_mean:.3f} ms (±{orig_std:.3f}), n={N_RUNS}\n")
        f.write(f"- fused(우리 C++, naive): {cpp_mean:.3f} ms (±{cpp_std:.3f}), n={N_RUNS}\n")
        f.write(f"- 배율: {cpp_mean / orig_mean:.2f}x\n")
    print("\n결과를 docs/backend_correctness_verification.md 에 추가했습니다.")


if __name__ == "__main__":
    main()
