"""패턴 매칭 로직 — 여기부터는 직접 구현하는 부분입니다.

1단계 목표: Conv 노드 하나를 보고 depthwise conv인지 판별하는 함수를 완성하세요.
"""

import onnx


def is_depthwise_conv(node: onnx.NodeProto) -> bool:
    """주어진 노드가 depthwise convolution인지 판별한다.

    TODO: 직접 구현하세요.

    힌트:
    - node.op_type이 "Conv"가 아니면 당연히 False
    - Conv 노드의 attribute 중 이름이 "group"인 것을 찾아야 합니다.
      node.attribute는 onnx.AttributeProto의 리스트이고,
      각 attribute는 .name과 .i(정수 값) 필드를 가집니다.
      (또는 onnx.helper.get_attribute_value(attr)로 값을 뽑아도 됩니다)
    - depthwise conv의 정의: group 수가 입력 채널 수와 같을 때.
      하지만 이 함수는 node 하나만 보고 판단해야 해서, 입력 채널 수를
      직접 알기 어려울 수 있습니다. 우선은 "group > 1이면 depthwise로
      본다"는 단순화된 기준으로 시작해보고, 나중에 더 정확하게
      판별하는 방법이 필요한지 스스로 판단해보세요.
    - group attribute가 아예 없는 Conv도 있습니다 (기본값 1, 즉
      일반 conv) — 이 경우도 처리해야 합니다.

    Args:
        node: onnx.NodeProto 하나

    Returns:
        depthwise conv이면 True, 아니면 False
    """
    # 초기 동민 ver.
    # if node.op_type != "Conv": #node.op_type의 종류: Conv, Clip, Add, ReduceMean, Reshape, Gemm
    #     return False
    # if not node.attribute: #속성이 없으면 안됌
    #     return False
    # pos_attr = False
    # for attr in node.attribute:
    #     if attr.name == "group": #group이 1이면 일반 conv
    #         pos_attr = True
    #         if attr.i == 1:
    #           return False
    #         break
    
    # return pos_attr#나머지 경우는 depthwise conv로 간주

    # 정석 ver.
    if node.op_type != "Conv":
        return False
    group = next((attr.i for attr in node.attribute if attr.name == "group"), 1)
    return group > 1

def _is_pointwise_conv(node: onnx.NodeProto) -> bool:
    if node.op_type != "Conv":
        return False
    kernel_shape = next((attr.ints for attr in node.attribute if attr.name == "kernel_shape"),None)
    group = next((attr.i for attr in node.attribute if attr.name == "group"), 1)
    return kernel_shape == [1,1] and group == 1

def match_inverted_residual(
    graph: onnx.GraphProto,
    producer_map: dict[str, onnx.NodeProto],
    consumer_map: dict[str, list[onnx.NodeProto]],
) -> list[dict]:
    """그래프에서 inverted residual 블록들을 전부 찾아 반환한다.

    TODO: 직접 구현하세요.

    접근 방법 힌트:
    1. graph.node를 순회하며 is_depthwise_conv(node)로 depthwise conv를
       전부 찾는다 — 이게 각 블록의 "앵커"입니다 (17개 나와야 정상).

    2. 앵커 하나마다, 뒤(forward)로 consumer_map을 따라가며:
       - depthwise의 output -> consumer는 Clip이어야 함
       - 그 Clip의 output -> consumer는 Conv(project)여야 함
       - project의 output -> consumer 중에 Add가 있으면 residual 있음
         (Type C), 없으면 residual 없음 (Type A/B)

    3. 앵커 하나마다, 앞(backward)으로 producer_map을 따라가며:
       - depthwise의 input[0] -> producer가 Clip인지 확인
       - 그 Clip의 input[0] -> producer가 Conv인지 확인
       - **주의**: 여기서 찾은 Conv가 진짜 이 블록의 expand layer인지
         확인하려면 추가 체크가 필요합니다. expand/project conv는
         항상 1x1(pointwise) conv라서, kernel_shape이 [1,1]이고
         group이 1인지 확인해야 합니다. (안 그러면 features.1의 경우
         stem conv를 expand로 잘못 착각하게 됩니다 — stem도 Clip 뒤에
         Conv가 있는 구조라서 겉보기엔 똑같아 보이거든요.)
       - 조건을 만족하면 expand 있는 블록 (Type B/C), 아니면 없는 블록
         (Type A)

    4. 2번, 3번 결과를 조합해서 Type A/B/C를 확정하고, 그 블록에
       속하는 노드들을 모아서 결과에 추가한다.

    Args:
        graph: onnx.GraphProto
        producer_map: graph_utils.build_producer_map()의 결과
        dict[str, onnx.NodeProto]
        consumer_map: graph_utils.build_consumer_map()의 결과
        dict[str, list[onnx.NodeProto]]

    Returns:
        매칭된 블록들의 리스트. 각 블록을 어떻게 표현할지(타입 라벨 +
        노드 리스트를 dict로 묶는 등)는 직접 설계해보세요.
        예: [{"type": "A", "nodes": [...]}, {"type": "C", "nodes": [...]}, ...]
    """
    blocks= []
    for dw_node in graph.node:
        if not is_depthwise_conv(dw_node):
            continue

        # --- 뒤(forward): depthwise -> Clip -> project Conv ---
        dw_clip = consumer_map.get(dw_node.output[0], [None])[0] # Clip 노드
        if dw_clip is None or dw_clip.op_type != "Clip":
            continue
        project = consumer_map.get(dw_clip.output[0], [None])[0] # project Conv 노드
        if project is None or project.op_type != "Conv":
            continue

        # --- 앞(backward): depthwise <- Clip <- exapand Conv ---
        expand_clip = producer_map.get(dw_node.input[0]) # Clip
        expand = None
        if expand_clip is not None and expand_clip.op_type == "Clip":
            candidate = producer_map.get(expand_clip.input[0])
            if candidate is not None and _is_pointwise_conv(candidate):
                expand = candidate

        # --- residual(Add) 있는지 ---
        # 이 블록이 "원래 받았던 입력"을 기억해둔다 (expand가 있으면 expand의
        # 입력, 없으면 depthwise의 입력) — Add가 진짜 이 블록 자신의
        # residual인지 확인할 때 이 값이랑 대조해야 한다.
        block_input = expand.input[0] if expand else dw_node.input[0]
        add = next(
            (
                c
                for c in consumer_map.get(project.output[0], [])
                if c.op_type == "Add" and block_input in c.input
            ),
            None,
        )

        block_nodes = ([expand, expand_clip] if expand else []) + [dw_node, dw_clip, project]
        if add:
            block_nodes.append(add)

        block_type = "C" if add else ("B" if expand else "A")
        blocks.append({"type": block_type, "nodes": block_nodes})

    return blocks
