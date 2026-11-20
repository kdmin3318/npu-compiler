"""특정 블록 하나를 '단독으로' 떼어내 unfused vs fused 비교.
synthetic 입력(해당 블록의 실제 입력 shape, 랜덤 값)을 사용해서
shape inference 문제를 피함.
"""

import ctypes
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

_ORT_DLL = Path(__file__).parent.parent / ".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"
ctypes.CDLL(str(_ORT_DLL))


def build_single_block_model(full_model_path: str, block_order_index: int, input_shape: list[int]):
    model = onnx.load(full_model_path)
    graph = model.graph
    fused_nodes = [n for n in graph.node if n.op_type == "Inverted_residual"]
    node = fused_nodes[block_order_index]
    print(f"block {block_order_index}: block_type=", end="")
    for attr in node.attribute:
        if attr.name == "block_type":
            print(attr.s.decode(), end=" ")
        if attr.name == "dw_stride":
            print(f"dw_stride={attr.i}", end="")
    print()

    needed_names = set(node.input)
    initializers = [init for init in graph.initializer if init.name in needed_names]
    block_input_name = node.input[0]

    new_input = onnx.helper.make_tensor_value_info(block_input_name, onnx.TensorProto.FLOAT, input_shape)
    new_output = onnx.helper.make_tensor_value_info(node.output[0], onnx.TensorProto.FLOAT, None)
    new_graph = onnx.helper.make_graph([node], "single_block", [new_input], [new_output], initializer=initializers)
    new_model = onnx.helper.make_model(new_graph, opset_imports=list(model.opset_import))
    new_model.ir_version = model.ir_version
    return new_model


def test_block(block_order_index: int, input_shape: list[int]) -> None:
    model = build_single_block_model("models/mobilenetv2_cpp.onnx", block_order_index, input_shape)
    onnx.save(model, "models/_debug_single_block.onnx")

    so_unfused = ort.SessionOptions()
    so_unfused.register_custom_ops_library("cpp_backend/inverted_residual_op_dnnl_unfused.dll")
    sess_unfused = ort.InferenceSession("models/_debug_single_block.onnx", so_unfused)

    so_fused = ort.SessionOptions()
    so_fused.register_custom_ops_library("cpp_backend/inverted_residual_op_dnnl_fused.dll")
    sess_fused = ort.InferenceSession("models/_debug_single_block.onnx", so_fused)

    input_name = sess_unfused.get_inputs()[0].name
    rng = np.random.default_rng(0)
    sample = rng.standard_normal(input_shape).astype(np.float32)

    out_unfused = sess_unfused.run(None, {input_name: sample})[0]
    out_fused = sess_fused.run(None, {input_name: sample})[0]

    diff = np.abs(out_unfused - out_fused)
    ok = np.allclose(out_unfused, out_fused, rtol=1e-3, atol=1e-4)
    print(f"  shape={out_unfused.shape} 최대차이={diff.max():.4f} allclose={ok}")
    if not ok:
        bad = np.argwhere(diff > 1e-2)
        print(f"  차이 큰 위치: {len(bad)}/{diff.size}")
        for idx in bad[:5]:
            c, h, w = idx[1], idx[2], idx[3]
            print(f"    [{c},{h},{w}]: unfused={out_unfused[0,c,h,w]:.4f} fused={out_fused[0,c,h,w]:.4f}")


def main() -> None:
    # block 0: A, stride1, 32ch 112x112 -> 이미 확인됨 (정확)
    # block 1: B, stride2, 16ch 112x112 -> 틀림 (확인됨)
    test_block(1, [1, 16, 112, 112])
    # block 2: 보통 C (residual, stride1, expand 있음) - 24ch 56x56 예상
    test_block(2, [1, 24, 56, 56])


if __name__ == "__main__":
    main()
