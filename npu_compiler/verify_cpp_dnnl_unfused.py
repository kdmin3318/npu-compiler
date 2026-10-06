"""unfused+oneDNN 커스텀 op(cpp_backend/inverted_residual_op_dnnl_unfused.dll)이
원본 모델과 수치적으로 동일한 결과를 내는지 검증한다. verify_cpp.py와 동일한
방식(5 seed, allclose)이고 대상 DLL만 다르다.
"""

import ctypes
from pathlib import Path

import numpy as np
import onnxruntime as ort

_ORT_DLL = Path(__file__).parent.parent / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
ctypes.CDLL(str(_ORT_DLL))


def main() -> None:
    results = []
    orig_sess = ort.InferenceSession("models/mobilenetv2.onnx")

    so = ort.SessionOptions()
    so.register_custom_ops_library("cpp_backend/inverted_residual_op_dnnl_unfused.dll")
    cpp_sess = ort.InferenceSession("models/mobilenetv2_cpp.onnx", so)

    all_ok = True
    for seed in range(5):
        rng = np.random.default_rng(seed)
        sample_input = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)

        orig_out = orig_sess.run(None, {"input": sample_input})[0]
        cpp_out = cpp_sess.run(None, {"input": sample_input})[0]

        diff = float(np.max(np.abs(orig_out - cpp_out)))
        ok = bool(np.allclose(orig_out, cpp_out, rtol=1e-3, atol=1e-4))
        all_ok = all_ok and ok
        results.append((seed, diff, ok))
        print(f"seed={seed}: 최대차이={diff:.2e} allclose={ok}")

    print(f"\n전체 결과: {'모두 통과' if all_ok else '일부 실패'}")


if __name__ == "__main__":
    main()
