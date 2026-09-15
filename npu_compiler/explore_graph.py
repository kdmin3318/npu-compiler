"""1단계 결과 확인용 테스트 스크립트 (boilerplate).

patterns.py의 is_depthwise_conv()를 구현한 뒤 이 스크립트를 돌려서
확인하세요.

실행: uv run python -m npu_compiler.explore_graph
"""

from pathlib import Path

from npu_compiler.graph_utils import load_graph, summarize
from npu_compiler.patterns import is_depthwise_conv

MODEL_PATH = Path(__file__).parent.parent / "models" / "mobilenetv2.onnx"


def main() -> None:
    graph = load_graph(str(MODEL_PATH))
    summarize(graph)

    print()
    print("Conv 노드별 depthwise 여부:")
    conv_nodes = [n for n in graph.node if n.op_type == "Conv"]
    depthwise_count = 0
    for node in conv_nodes:
        is_dw = is_depthwise_conv(node)
        if is_dw:
            depthwise_count += 1
        print(f"  {node.name or '(unnamed)'}: depthwise={is_dw}")

    print()
    print(f"전체 Conv 노드 수: {len(conv_nodes)}")
    print(f"Depthwise Conv 노드 수: {depthwise_count}")
    print("(MobileNetV2는 보통 17~18개의 depthwise conv를 가집니다 - 비교해보세요)")


if __name__ == "__main__":
    main()
