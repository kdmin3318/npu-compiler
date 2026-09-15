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
