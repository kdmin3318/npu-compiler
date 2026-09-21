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


def build_fused_node(block: dict) -> onnx.NodeProto:
    """블록 하나(match_inverted_residual의 결과 원소 하나)를 대표하는
    커스텀 fused 노드 하나를 만든다.

    TODO: 직접 구현하세요.

    접근 방법 힌트:
    - `block["nodes"]`는 이 블록에 속한 노드들의 **순서 있는** 리스트입니다.
    - 이 블록의 진짜 입력(block_input) = **첫 번째 노드의 input[0]**
      (Type A/B/C 상관없이 첫 노드가 뭐든 그 노드의 input[0]이 곧 block_input
      입니다 — match_inverted_residual에서 이미 확인한 사실을 재사용하는 것)
    - 이 블록의 진짜 출력 = **마지막 노드의 output[0]**
      (Type C면 마지막이 Add, 아니면 project Conv)
    - weight/bias 입력들: `block["nodes"]` 중 op_type이 "Conv"인 것들의
      `input[1:]` (즉 데이터 입력인 input[0]은 빼고)를 순서대로 모으면 됩니다.
    - 최종 `inputs` = `[block_input]` + 모든 weight/bias 이름들
    - **입출력 텐서 이름을 원래 이름 그대로 재사용하면**(새로 이름을 짓지
      않고), 이 fused 노드 뒤에 연결된 다른 노드들(다음 블록, 또는 residual
      로 이 출력을 갖다 쓰는 나중 블록)이 아무것도 안 바꿔도 자동으로 계속
      연결됩니다 — 그래프 재작성(2번 함수)을 훨씬 쉽게 만들어주는 포인트
      입니다.
    - `onnx.helper.make_node(op_type, inputs, outputs, name=..., domain=...)`
      로 노드를 만듭니다. 표준 op이 아니니 domain을 커스텀 값(예:
      "com.npu_compiler")으로 지정하세요.
    - `block["type"]`도 나중에 확인하기 편하게 attribute로 같이 넣어두면
      좋습니다 (`make_node`에 정의 안 된 키워드 인자를 추가로 넘기면
      자동으로 attribute가 됩니다, 예: `make_node(..., block_type=block["type"])`).

    Args:
        block: {"type": "A"/"B"/"C", "nodes": [...]} 형태의 dict

    Returns:
        onnx.NodeProto — 이 블록을 대표하는 커스텀 fused 노드 하나
    """
    weights = []
    for node in block["nodes"]:
        if node.op_type == "Conv":
            weights.extend(node.input[1:]) # weight/bias만 모음
    fused_node = onnx.helper.make_node(
        op_type = "Inverted_residual",
        inputs = [block["nodes"][0].input[0]] + weights,
        outputs = [block["nodes"][-1].output[0]], 
        domain = "com.npu_compiler",
    )
    return fused_node


def rewrite_graph(graph: onnx.GraphProto, blocks: list[dict]) -> onnx.GraphProto:
    """매칭된 블록들을 fused 노드로 교체한 새 그래프를 만들어 반환한다.

    TODO: 직접 구현하세요.

    접근 방법 힌트:
    1. 각 블록마다 build_fused_node(block)으로 fused 노드를 미리 만들어
       둔다 (예: block마다 하나씩, 리스트나 dict로 준비).

    2. "어떤 노드가 어떤 블록에 속하는지" 빠르게 찾을 수 있는 인덱스를
       만든다. NodeProto는 파이썬 딕셔너리 키로 바로 못 쓰니(해시 불가),
       id(node)(객체의 메모리 주소, 정수라서 해시 가능)를 키로 쓰세요.
       예: {id(node): block_index
            for block_index, block in enumerate(blocks)
            for node in block["nodes"]}

    3. graph.node를 원래 순서대로 순회하면서 새 노드 리스트를 만든다:
       - 이 노드가 어떤 블록에도 안 속하면 -> 그대로 새 리스트에 추가
       - 이 노드가 어떤 블록에 속하면 -> 그 블록의 fused 노드를 **딱 한
         번만** 새 리스트에 추가하고, 같은 블록의 나머지 노드들은
         건너뛴다 (이미 그 블록의 fused 노드를 추가했으면 또 추가하면
         안 됨 — 어떤 블록을 이미 처리했는지 추적하는 집합(set)이 필요할
         수 있습니다).

    4. onnx.helper.make_graph()로 새 GraphProto를 만든다. graph.initializer,
       graph.input, graph.output은 안 바뀌었으니 그대로 재사용하고,
       node만 3번에서 만든 새 리스트로 넣으면 됩니다. graph.node에
       직접 대입(`graph.node = ...`)은 안 되니 주의하세요 (반복 필드라서).

    Args:
        graph: 원본 onnx.GraphProto (이 함수 안에서 직접 수정하지 않는다)
        blocks: match_inverted_residual()의 결과

    Returns:
        onnx.GraphProto — 블록들이 fused 노드로 교체된 새 그래프
    """
    fused_nodes = [build_fused_node(block) for block in blocks]

    node_to_block_index = {
        id(node): block_index
        for block_index, block in enumerate(blocks)
        for node in block["nodes"]
    }

    new_nodes = []
    inserted_block_indices = set()
    for node in graph.node:
        block_index = node_to_block_index.get(id(node))
        if block_index is None:
            new_nodes.append(node)
        elif block_index not in inserted_block_indices:
            new_nodes.append(fused_nodes[block_index])
            inserted_block_indices.add(block_index)

    return onnx.helper.make_graph(
        new_nodes,
        graph.name,
        graph.input,
        graph.output,
        initializer=graph.initializer,
    )
