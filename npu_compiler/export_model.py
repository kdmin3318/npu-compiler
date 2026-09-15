"""MobileNetV2를 ONNX로 export하는 스크립트 (boilerplate).

실행: uv run python -m npu_compiler.export_model
"""

from pathlib import Path

import torch
import torchvision

MODEL_PATH = Path(__file__).parent.parent / "models" / "mobilenetv2.onnx"


def main() -> None:
    model = torchvision.models.mobilenet_v2(
        weights=torchvision.models.MobileNet_V2_Weights.DEFAULT
    )
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(MODEL_PATH),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )
    print(f"Exported MobileNetV2 to {MODEL_PATH}")


if __name__ == "__main__":
    main()
