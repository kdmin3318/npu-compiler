"""C++ 커스텀 op(cpp_backend/inverted_residual_op.cpp)용 fused 모델을
준비한다. domain은 build_fused_node()의 기본값(com.npu_compiler)을
그대로 쓰고, Type A 블록은 없는 expand 자리를 빈 문자열("")로 채워서
7개 입력 자리를 항상 유지한다 (ONNX의 "중간 optional 입력" 표준 방식 -
onnxruntime_extensions의 PyOp과 달리, 진짜 OrtCustomOp C API는 이 방식을
정식으로 지원한다).
"""

import onnx

from npu_compiler.graph_utils import build_consumer_map, build_producer_map, load_graph
from npu_compiler.patterns import build_fused_node, match_inverted_residual


def build_cpp_model(onnx_path: str) -> onnx.ModelProto:
    graph = load_graph(onnx_path)
    producer_map = build_producer_map(graph)
    consumer_map = build_consumer_map(graph)
    blocks = match_inverted_residual(graph, producer_map, consumer_map)

    fused_nodes = []
    for b in blocks:
        node = build_fused_node(b)  # domain 기본값(com.npu_compiler) 그대로
        if b["type"] == "A":
            block_input = node.input[0]
            padded_inputs = [block_input, "", ""] + list(node.input[1:])
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
            onnx.helper.make_opsetid("com.npu_compiler", 1),
        ],
    )
    model.ir_version = 8
    return model


if __name__ == "__main__":
    model = build_cpp_model("models/mobilenetv2.onnx")
    onnx.save(model, "models/mobilenetv2_cpp.onnx")
    print("저장 완료: models/mobilenetv2_cpp.onnx")
