"""2차 성능 측정: 원본(unfused, MLAS) vs naive fused(C++) vs unfused+oneDNN
세 가지 wall-clock 비교. "변인 통제" 베이스라인 구축 1단계 - fusion 없이
oneDNN만 적용했을 때 naive 대비 얼마나 빨라지는지 확인하기 위함.
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

    orig_sess = ort.InferenceSession("models/mobilenetv2.onnx")
    orig_times = _time_session(orig_sess, sample_input)

    so_naive = ort.SessionOptions()
    so_naive.register_custom_ops_library("cpp_backend/inverted_residual_op.dll")
    naive_sess = ort.InferenceSession("models/mobilenetv2_cpp.onnx", so_naive)
    naive_times = _time_session(naive_sess, sample_input)

    so_dnnl = ort.SessionOptions()
    so_dnnl.register_custom_ops_library("cpp_backend/inverted_residual_op_dnnl_unfused.dll")
    dnnl_sess = ort.InferenceSession("models/mobilenetv2_cpp.onnx", so_dnnl)
    dnnl_times = _time_session(dnnl_sess, sample_input)

    results = {
        "원본 (unfused, MLAS)": orig_times,
        "fused, naive C++": naive_times,
        "unfused, oneDNN": dnnl_times,
    }

    print(f"{'구성':<24}{'평균(ms)':>12}{'표준편차':>12}{'원본 대비':>12}")
    orig_mean = np.mean(orig_times) * 1000
    for name, times in results.items():
        mean_ms = np.mean(times) * 1000
        std_ms = np.std(times) * 1000
        ratio = mean_ms / orig_mean
        print(f"{name:<24}{mean_ms:>12.3f}{std_ms:>12.3f}{ratio:>11.2f}x")


if __name__ == "__main__":
    main()
