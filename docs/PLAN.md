# 프로젝트 계획: MobileNet Inverted Residual Block Fusion

## 배경 및 가설

MobileNet 계열의 inverted residual 블록(1x1 expand conv → depthwise 3x3 conv → 1x1
project conv → [옵션] residual add)은 ONNX 그래프에서 개별 연산 노드로 분리되어
표현되고, 별도의 컴파일러 최적화 없이는 각 단계가 따로 실행된다.

depthwise conv는 연산 강도(arithmetic intensity)가 낮아 memory-bound 특성을 가진다.
즉 단계별로 따로 실행하면 중간 activation을 매번 DRAM에 write/read 해야 하는데,
이 세 단계를 하나의 fused block으로 묶어 on-chip buffer(SRAM/scratchpad)에서만
데이터를 유지하면 DRAM 트래픽이 줄어들어 실제로 속도가 빨라질 것이라는 가설에서
출발한다.

## 현재 스코프

- 실제 NPU 하드웨어는 보유하고 있지 않음 (랩실 보유 여부 확인 예정 — 미결 사항 참고)
- 목표는 **그래프 레벨 최적화**: ONNX 그래프에서 inverted residual 패턴을 자동으로
  인식하고 하나의 fused op으로 재작성하는 것
- 성능 검증은 실측 대신 분석적 메모리 트래픽 모델로 시작하고, 여유가 있으면
  오픈소스 NPU 시뮬레이터(예: ARM Vela, SCALE-Sim) 연동을 검토한다

## 사용 언어/도구

- **Python** (`onnx`, `onnxruntime`, `netron`)
- 근거: 그래프 레벨 분석/재작성은 업계·연구 관행상 Python이 표준(TVM, MLIR 등도
  프론트엔드/패스 스크립팅은 Python). 실제 커널 코드 생성이나 런타임(C++) 통합
  단계까지 가면 C++가 필요해질 수 있으나, 현재 스코프에서는 불필요

## 진행 방식

- 컴파일러를 처음 다뤄보는 프로젝트이므로, 핵심 로직(그래프 순회, 패턴 매칭,
  fusion 재작성)은 직접 구현하며 학습한다
- Claude는 힌트 제공, API 사용법 설명, 코드 리뷰, 다음 단계 방향 제시 역할을 맡는다

## 마일스톤

0. **환경 준비**: Python 환경 + `onnx`/`onnxruntime`/`netron` 설치, MobileNetV2
   ONNX 모델 export (torchvision → `torch.onnx.export`)
1. **그래프 구조 파악**: `onnx.load()`로 노드 순회, `Conv`의 `group` attribute로
   depthwise 여부 판별
2. **패턴 매칭**: inverted residual 시퀀스 탐지 로직 작성
   (1x1 conv → activation → depthwise conv → activation → 1x1 conv → [옵션] add).
   BN fold 여부, activation 종류(Relu / Clip(Relu6) / 없음) 등 variation 처리
3. **그래프 재작성 (fusion)**: 매칭된 서브그래프를 custom op(예:
   `com.mylab.FusedInvertedResidual`)으로 치환, 원본 weight는 attribute로 보존,
   `onnx.checker.check_model()`로 검증
4. **성능 가설 검증**: fusion 전/후 DRAM read/write 바이트 수를 tensor shape 기반
   으로 분석적으로 계산해 비교, 필요시 시뮬레이터 연동
5. **정리 및 문서화**

## 미결 사항

- 랩실 NPU 하드웨어/시뮬레이터 보유 여부 확인 필요
- 참고할 특정 논문은 없음 — 자체 가설로 진행하기로 결정 (2026-09-14)
