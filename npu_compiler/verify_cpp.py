"""C++ 커스텀 op(cpp_backend/inverted_residual_op.dll)이 실제로 실행되고,
원본 모델과 수치적으로 동일한 결과를 내는지 검증한다.

주의(Windows): Windows에 내장된 구버전 onnxruntime.dll(System32,
Windows AI 플랫폼용)이 venv의 onnxruntime.dll보다 DLL 검색 순서상
먼저 잡혀서, 그냥 두면 버전 충돌(ORT_API_VERSION mismatch)이 난다.
`os.add_dll_directory()`로도 안 풀려서(이미 로드된 DLL은 이름으로
캐시돼 재사용됨), ctypes.CDLL로 올바른 버전을 절대경로로 먼저
강제 로드해서 그 캐시를 선점하는 방식으로 우회한다.
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
    so.register_custom_ops_library("cpp_backend/inverted_residual_op.dll")
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

    with open("docs/backend_correctness_verification.md", "a", encoding="utf-8") as f:
        f.write("\n## C++ 커스텀 op(실제 실행) 검증 결과\n\n")
        f.write("PyOp이 아니라 진짜 컴파일된 C++ 커스텀 op(`cpp_backend/inverted_residual_op.dll`)")
        f.write("로 실행한 결과.\n\n")
        f.write("| seed | 최대 차이 | allclose |\n|---|---|---|\n")
        for seed, diff, ok in results:
            f.write(f"| {seed} | {diff:.2e} | {ok} |\n")
        f.write(f"\n**전체: {'모두 통과' if all_ok else '일부 실패'}**\n")
    print("docs/backend_correctness_verification.md 에 결과 추가했습니다.")


if __name__ == "__main__":
    main()
