"""naive 타일링 버전(cpp_backend/inverted_residual_op_naive_tiled.dll) 정확성 검증."""

import ctypes
from pathlib import Path

import numpy as np
import onnxruntime as ort

_ORT_DLL = Path(__file__).parent.parent / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
ctypes.CDLL(str(_ORT_DLL))


def main() -> None:
    orig_sess = ort.InferenceSession("models/mobilenetv2.onnx")

    so = ort.SessionOptions()
    so.register_custom_ops_library("cpp_backend/inverted_residual_op_naive_tiled.dll")
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
        print(f"seed={seed}: 최대차이={diff:.2e} allclose={ok}")

    print(f"\n전체 결과: {'모두 통과' if all_ok else '일부 실패'}")


if __name__ == "__main__":
    main()
