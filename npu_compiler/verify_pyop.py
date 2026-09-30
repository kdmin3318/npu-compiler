"""PyOp(onnxruntime-extensions)로 fused 모델을 실제로 실행해서, 원본
모델과 출력이 같은지(correctness) 검증한다. 속도 측정 목적 아님 —
Python 콜백 오버헤드 때문에 여기서 잰 시간은 의미 없다.
"""

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime_extensions import PyCustomOpDef, get_library_path, onnx_op

from npu_compiler.graph_utils import build_consumer_map, build_producer_map, load_graph
from npu_compiler.patterns import build_fused_node, match_inverted_residual

VERIFY_DOMAIN = "ai.onnx.contrib"  # onnxruntime-extensions의 PyOp가 고정으로 쓰는 domain


def _relu6(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 6.0)


def _conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int, groups: int) -> np.ndarray:
    """정확성 검증용 naive 참조 구현. 속도는 신경 안 씀."""
    n, cin, h, w_dim = x.shape
    cout, _, kh, kw = w.shape
    pad = kh // 2
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    hp, wp = xp.shape[2], xp.shape[3]
    hout = (hp - kh) // stride + 1
    wout = (wp - kw) // stride + 1
    out = np.zeros((n, cout, hout, wout), dtype=np.float32)
    out_per_group = cout // groups
    in_per_group = cin // groups
    for g in range(groups):
        xg = xp[:, g * in_per_group : (g + 1) * in_per_group]
        wg = w[g * out_per_group : (g + 1) * out_per_group]
        for oh in range(hout):
            for ow in range(wout):
                patch = xg[:, :, oh * stride : oh * stride + kh, ow * stride : ow * stride + kw]
                out[:, g * out_per_group : (g + 1) * out_per_group, oh, ow] = np.einsum(
                    "ncij,ocij->no", patch, wg
                )
    out += b.reshape(1, -1, 1, 1)
    return out


@onnx_op(
    op_type="Inverted_residual",
    inputs=[PyCustomOpDef.dt_float] * 7,
    outputs=[PyCustomOpDef.dt_float],
    attrs={"block_type": PyCustomOpDef.dt_string, "dw_stride": PyCustomOpDef.dt_int64},
)
def inverted_residual_op(*inputs, block_type, dw_stride):
    block_input = inputs[0]
    if block_type == "A":
        # inputs[1], inputs[2]는 더미(padding용, 값 안 씀) - 실제 값은 3번부터
        dw_w, dw_b, proj_w, proj_b = inputs[3:7]
        x = _relu6(_conv2d(block_input, dw_w, dw_b, stride=dw_stride, groups=dw_w.shape[0]))
        x = _conv2d(x, proj_w, proj_b, stride=1, groups=1)
    else:
        expand_w, expand_b, dw_w, dw_b, proj_w, proj_b = inputs[1:7]
        x = _relu6(_conv2d(block_input, expand_w, expand_b, stride=1, groups=1))
        x = _relu6(_conv2d(x, dw_w, dw_b, stride=dw_stride, groups=dw_w.shape[0]))
        x = _conv2d(x, proj_w, proj_b, stride=1, groups=1)
        if block_type == "C":
            x = x + block_input
    return x.astype(np.float32)


def build_verification_model(onnx_path: str) -> onnx.ModelProto:
    graph = load_graph(onnx_path)
    producer_map = build_producer_map(graph)
    consumer_map = build_consumer_map(graph)
    blocks = match_inverted_residual(graph, producer_map, consumer_map)

    fused_nodes = []
    for b in blocks:
        node = build_fused_node(b, domain=VERIFY_DOMAIN)
        if b["type"] == "A":
            # onnxruntime-extensions는 optional/variadic 입력을 지원하지
            # 않아서 (schema가 고정 개수), 항상 7개를 맞추려고 "없는
            # expand" 자리를 block_input으로 채운 더미 값을 넣는다.
            # inverted_residual_op()는 block_type=="A"일 때 이 두 값을
            # 아예 안 읽으므로 값 자체는 의미 없다 (검증 전용 우회).
            block_input = node.input[0]
            padded_inputs = [node.input[0], block_input, block_input] + list(node.input[1:])
            node = onnx.helper.make_node(
                op_type=node.op_type,
                inputs=padded_inputs,
                outputs=list(node.output),
                domain=node.domain,
                block_type="A",
                dw_stride=next(a.i for a in node.attribute if a.name == "dw_stride"),
            )
        fused_nodes.append(node)
    node_to_block_index = {
        id(node): i for i, block in enumerate(blocks) for node in block["nodes"]
    }
    new_nodes = []
    inserted = set()
    for node in graph.node:
        idx = node_to_block_index.get(id(node))
        if idx is None:
            new_nodes.append(node)
        elif idx not in inserted:
            new_nodes.append(fused_nodes[idx])
            inserted.add(idx)

    new_graph = onnx.helper.make_graph(
        new_nodes, graph.name, graph.input, graph.output, initializer=graph.initializer
    )
    model = onnx.helper.make_model(
        new_graph,
        opset_imports=[
            onnx.helper.make_opsetid("", 18),
            onnx.helper.make_opsetid(VERIFY_DOMAIN, 1),
        ],
    )
    model.ir_version = 8
    return model


def _with_extra_outputs(model: onnx.ModelProto, names: list[str]) -> onnx.ModelProto:
    new_model = onnx.ModelProto()
    new_model.CopyFrom(model)
    value_info_map = {
        vi.name: vi for vi in list(new_model.graph.value_info) + list(new_model.graph.input)
    }
    existing = {o.name for o in new_model.graph.output}
    for name in names:
        if name in existing:
            continue
        if name in value_info_map:
            vi = onnx.ValueInfoProto()
            vi.CopyFrom(value_info_map[name])
        else:
            vi = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
        new_model.graph.output.append(vi)
        existing.add(name)
    return new_model


def main() -> None:
    graph = load_graph("models/mobilenetv2.onnx")
    producer_map = build_producer_map(graph)
    consumer_map = build_consumer_map(graph)
    blocks = match_inverted_residual(graph, producer_map, consumer_map)

    # 블록 경계 텐서 이름들 수집 (block_input, block_output) - 원본/fused가
    # 이름을 그대로 공유하므로 같은 이름으로 두 모델의 값을 비교할 수 있다.
    boundary_names = sorted(
        {b["nodes"][0].input[0] for b in blocks} | {b["nodes"][-1].output[0] for b in blocks}
    )

    original_model = onnx.load("models/mobilenetv2.onnx")
    final_output_name = original_model.graph.output[0].name

    original_debug_model = _with_extra_outputs(original_model, boundary_names)
    original_sess = ort.InferenceSession(original_debug_model.SerializeToString())
    original_output_names = [o.name for o in original_debug_model.graph.output]

    fused_model = build_verification_model("models/mobilenetv2.onnx")
    fused_debug_model = _with_extra_outputs(fused_model, boundary_names)
    onnx.save(fused_debug_model, "models/mobilenetv2_verify.onnx")
    so = ort.SessionOptions()
    so.register_custom_ops_library(get_library_path())
    fused_sess = ort.InferenceSession("models/mobilenetv2_verify.onnx", so)
    fused_output_names = [o.name for o in fused_debug_model.graph.output]

    report_lines = [
        "# PyOp 기반 fused 모델 correctness 검증 결과",
        "",
        "목적: fused 커스텀 op(`Inverted_residual`)이 원본(unfused) 그래프와",
        "수치적으로 동일한 결과를 계산하는지 검증. **속도 측정 아님** —",
        "PyOp은 Python 콜백 오버헤드가 있어 성능 지표로는 못 씀.",
        "",
        "| seed | 최종 출력 최대 차이 | 최종 출력 allclose | 블록 경계 전체 통과 |",
        "|---|---|---|---|",
    ]
    all_seeds_ok = True

    for seed in range(5):
        rng = np.random.default_rng(seed)
        sample_input = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)

        original_outputs = dict(
            zip(original_output_names, original_sess.run(None, {"input": sample_input}))
        )
        fused_outputs = dict(
            zip(fused_output_names, fused_sess.run(None, {"input": sample_input}))
        )

        final_diff = float(
            np.max(np.abs(original_outputs[final_output_name] - fused_outputs[final_output_name]))
        )
        final_ok = bool(
            np.allclose(
                original_outputs[final_output_name],
                fused_outputs[final_output_name],
                rtol=1e-3,
                atol=1e-4,
            )
        )

        block_all_ok = True
        for name in boundary_names:
            o, f = original_outputs[name], fused_outputs[name]
            if not np.allclose(o, f, rtol=1e-3, atol=1e-4):
                block_all_ok = False

        all_seeds_ok = all_seeds_ok and final_ok and block_all_ok
        report_lines.append(f"| {seed} | {final_diff:.2e} | {final_ok} | {block_all_ok} |")
        print(
            f"seed={seed}: 최종출력 최대차이={final_diff:.2e} allclose={final_ok}, "
            f"블록경계({len(boundary_names)}개) 전체통과={block_all_ok}"
        )

    report_lines += [
        "",
        f"**5개 시드 전체 결과: {'모두 통과' if all_seeds_ok else '일부 실패 - 확인 필요'}**",
        "",
        f"- 검증한 블록 경계 텐서 수: {len(boundary_names)}개 (17개 블록의 입력/출력, 중복 제거)",
        "- 허용 오차: rtol=1e-3, atol=1e-4 (float32 연산 순서 차이에 의한 반올림 오차 수준)",
    ]
    with open("docs/backend_correctness_verification.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print("\n결과를 docs/backend_correctness_verification.md 에 저장했습니다.")


if __name__ == "__main__":
    main()
