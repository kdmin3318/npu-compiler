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
