"""milestone 4 — fusion으로 절약되는 DRAM 트래픽을 분석적으로 추정한다.

여기부터는 직접 구현하는 부분입니다.
"""

import onnx

from npu_compiler.graph_utils import tensor_bytes


def estimate_block_traffic(
    block: dict,
    value_info_map: dict[str, onnx.ValueInfoProto],
) -> dict:
    """블록 하나에 대해, fusion 전/후 DRAM 트래픽(byte)을 계산한다.

    TODO: 직접 구현하세요.

    배경 (이미 정리했던 계산 방법):
    블록이 n1 -> n2 -> ... -> nk 순서라고 할 때, 텐서 흐름은
    block_input -> t1(n1 출력) -> t2(n2 출력) -> ... -> t_{k-1}(n_{k-1} 출력)
    -> block_output(nk 출력) 입니다.

    - fusion 안 했을 때: block_input 읽기(1회) + 중간 텐서(t1..t_{k-1}) 각각
      "만든 노드가 씀 + 다음 노드가 읽음"으로 2회씩 + block_output 쓰기(1회)
    - fusion 했을 때: block_input 읽기(1회) + block_output 쓰기(1회)만
      (중간 텐서는 전부 온칩에만 머물러서 DRAM에 안 나감)

    접근 방법 힌트:
    - block_input 텐서 이름 = block["nodes"][0].input[0]
    - block_output 텐서 이름 = block["nodes"][-1].output[0]
    - 중간(intermediate) 텐서 이름들 = block["nodes"]에서 **마지막 노드를
      제외한** 나머지 노드들 각각의 output[0]
    - 텐서 하나의 byte 크기는 graph_utils.tensor_bytes(이름, value_info_map)
      로 구하면 됩니다.
    - unfused_bytes = tensor_bytes(block_input) + 2 * sum(중간 텐서들의
      byte) + tensor_bytes(block_output)
    - fused_bytes = tensor_bytes(block_input) + tensor_bytes(block_output)
    - saved_bytes = unfused_bytes - fused_bytes

    (참고: Type C처럼 residual이 있으면 block_input을 Add에서 한 번 더
    읽어야 하니 unfused일 때 실제로는 조금 더 많이 읽힙니다. 처음엔 이건
    무시하고 기본 버전부터 만들고, 여유 되면 나중에 다듬어보세요.)

    Args:
        block: match_inverted_residual()의 결과 원소 하나
        value_info_map: graph_utils.build_value_info_map()의 결과

    Returns:
        {"unfused_bytes": int, "fused_bytes": int, "saved_bytes": int}
        형태의 dict (원하면 필드 더 추가해도 됩니다)
    """
    block_input = block["nodes"][0].input[0]
    block_output = block["nodes"][-1].output[0]
    intermediates = [n.output[0] for n in block["nodes"][:-1]]

    intermediate_bytes = sum(tensor_bytes(t, value_info_map) for t in intermediates)

    unfused_bytes = (
        tensor_bytes(block_input, value_info_map)
        + 2 * intermediate_bytes
        + tensor_bytes(block_output, value_info_map)
    )
    fused_bytes = tensor_bytes(block_input, value_info_map) + tensor_bytes(
        block_output, value_info_map
    )

    return {
        "unfused_bytes": unfused_bytes,
        "fused_bytes": fused_bytes,
        "saved_bytes": unfused_bytes - fused_bytes,
    }
