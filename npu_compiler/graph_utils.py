"""ONNX 그래프 로딩/요약 유틸 (boilerplate).

패턴 매칭 로직이 아니라 순수 I/O 및 정보 출력용 헬퍼입니다.
"""

from collections import Counter

import onnx


def load_graph(onnx_path: str) -> onnx.GraphProto:
    """ONNX 모델 파일을 읽어서 GraphProto를 반환한다."""
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    return model.graph


def summarize(graph: onnx.GraphProto) -> None:
    """그래프의 노드 개수를 op_type별로 요약 출력한다."""
    counts = Counter(node.op_type for node in graph.node)
    print(f"Total nodes: {len(graph.node)}")
    for op_type, count in counts.most_common():
        print(f"  {op_type}: {count}")


def build_producer_map(graph: onnx.GraphProto) -> dict[str, onnx.NodeProto]:
    """텐서 이름 -> 그 텐서를 output으로 만들어낸 노드.

    "이 노드의 입력이 어디서 왔는지" 역추적할 때 사용한다.
    (initializer로 주어지는 weight/bias 텐서는 어떤 노드의 output도
    아니므로 이 맵에 없다 — 역추적하다가 못 찾으면 그래프의 진짜
    입력이거나 weight라는 뜻이다.)
    """
    producer: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for output_name in node.output:
            producer[output_name] = node
    return producer


def build_consumer_map(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    """텐서 이름 -> 그 텐서를 input으로 쓰는 노드들(리스트, 여러 개일 수 있음).

    "이 노드의 출력을 누가 쓰는지" 순추적할 때 사용한다.
    """
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)
    return consumers
