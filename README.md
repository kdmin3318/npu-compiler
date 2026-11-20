# npu-compiler

MobileNetV2의 **inverted residual 블록을 ONNX 그래프 수준에서 자동으로 찾아 하나의 커스텀 op으로 fusion**하고,
"fusion으로 중간 activation의 DRAM 왕복을 없애면 빨라진다"는 가설을 C++ 백엔드로 직접 검증해 본 프로젝트입니다.

> **한 줄 결론**: 그래프 재작성과 fused 커널(halo 타일링 포함)은 모든 구현에서 원본과 수치적으로 일치했습니다.
> 하지만 이 CPU(Intel Core Ultra 7 258V)에서는 fusion으로 아낄 수 있는 메모리 시간의 **상한이 전체 실행 시간의 약 2%** 에
> 불과했고, 타일링 오버헤드가 그보다 커서 속도 향상은 관측되지 않았습니다. 가설은 반증되지 않았고,
> 메모리 시간이 전체의 큰 비중을 차지하는 타깃(작은 on-chip 메모리를 가진 NPU/MCU)에서만 검증할 수 있는 문제로 결론지었습니다.

---

## 1. 배경과 가설

Inverted residual 블록은 `1x1 expand conv → ReLU6 → 3x3 depthwise conv → ReLU6 → 1x1 project conv → [residual add]` 구조입니다.
ONNX 그래프에서는 이것이 개별 노드로 쪼개져 있어서, 단계마다 중간 결과(expand 출력은 입력의 최대 6배 크기)를 메모리에 썼다가 다시 읽습니다.

depthwise conv는 연산 강도(arithmetic intensity)가 낮아 memory-bound가 되기 쉽습니다. 그래서 세 단계를 하나로 묶고
타일 단위로 on-chip 버퍼 안에서만 처리하면 DRAM 트래픽이 줄어 빨라질 것이라는 가설에서 출발했습니다.
([docs/PLAN.md](docs/PLAN.md))

## 2. 관련 연구와 이 프로젝트의 위치

구현에 들어가기 전에, 같은 아이디어가 이미 있는지부터 찾아봤습니다.

| 연구 | 내용 | 이 프로젝트와의 관계 |
|---|---|---|
| Lavin, *On the Efficiency of Convolutional Neural Networks* ([arXiv:2404.03617](https://arxiv.org/abs/2404.03617), 2024) | inverted residual 블록 전체를 하나의 커널로 묶어 중간 activation을 DRAM에 쓰지 않는 **Block Fusion**. ConvFirst 모델이 같은 정확도에서 ConvNeXt(PyTorch Inductor) 대비 약 4배 빠름 (NVIDIA A5000, fp16, batch=128) | 아이디어 자체는 같음. 차이는 두 가지: ① 특정 블록에 맞춰 손으로 짠 커널이 아니라, **그래프를 순회해 패턴을 자동 탐지하는 범용 컴파일러 패스**로 구현 ② GPU·대량 배치가 아니라 **CPU·batch=1 온디바이스 조건**에서 검증 |
| Wu & Huang, *Data Locality Optimization of Depthwise Separable Convolutions for CNN Inference Accelerators* ([DATE 2019](https://doi.org/10.23919/DATE.2019.8715097)) | depthwise + pointwise fusion을 타일링으로 구현하고 DRAMSim2로 DRAM 트래픽 감소를 측정 | halo 타일링 커널 설계의 참고 자료 |
| *ShortcutFusion: From TensorFlow to FPGA-based Accelerator with Reuse-aware Memory Allocation for Shortcut Data* ([arXiv:2106.08167](https://arxiv.org/abs/2106.08167), IEEE TCAS-I) | residual(shortcut) 데이터를 on-chip에서 재사용하도록 메모리를 배치해 DRAM 접근을 줄임 | Type C 블록에서 residual 입력을 다시 읽지 않는 트래픽 계산의 근거. Lavin 논문은 프리프린트라 동료 심사를 거친 이 논문과 DATE 2019로 방향을 교차 확인 |
| *Fused-Layer CNNs for Memory-Efficient Inference on Microcontrollers* ([OpenReview](https://openreview.net/forum?id=2O8qbyxH6X)) | 공간 타일링으로 중간 feature map을 작은 메모리 안에서 처리 | 타일 단위 실행이 효과를 보는 조건(작은 on-chip 메모리)의 참고 |
| MCUNetV2 / TinyEngine (MIT HAN Lab) | 해상도가 큰 앞쪽 단계에만 patch 기반 실행을 적용하고, 세 conv를 하나의 루프 안에서 처리하는 전용 커널 사용 | 회고(7장)에서 "범용 라이브러리를 타일마다 호출하는 방식"과 대비 |
| *ExecuTorch* ([arXiv:2605.08195](https://arxiv.org/pdf/2605.08195)) | PyTorch 온디바이스 배포 스택. 그래프 변환 → Backend Delegate Interface(4.4절) → 실행 | 이 프로젝트의 그래프 패스는 delegate 앞단(partition/패턴 매칭), C++ 커스텀 op는 delegate가 하는 커널 실행에 해당. 다음 공부 방향 |

정리하면, 이 프로젝트는 새 기법을 발명한 것이 아니라 **알려진 기법(Block Fusion)을 임의의 ONNX 모델에 자동으로 적용하는 컴파일러 패스로 직접 구현하고, 다른 조건(CPU, batch=1)에서 효과가 재현되는지 검증**하는 것을 목표로 했습니다.

## 3. 파이프라인

```
torchvision MobileNetV2
   │  export_model.py            (torch.onnx.export, opset 18)
   ▼
ONNX 그래프 ── explore_graph.py   (Conv의 group으로 depthwise 판별: 52개 Conv 중 17개)
   │  patterns.py                (producer/consumer 맵 기반 패턴 매칭 → 17개 블록, Type A/B/C)
   ▼
fused 그래프 (com.npu_compiler::Inverted_residual × 17)
   │  estimate.py                (tensor shape 기반 분석적 DRAM 트래픽 추정)
   ▼
실행 백엔드
   ├─ PyOp (onnxruntime-extensions)      ── 정확성 검증 전용
   ├─ C++ naive                          ── OrtCustomOp, 3중 for문
   ├─ C++ oneDNN unfused                 ── 같은 라이브러리 베이스라인 (변인 통제)
   ├─ C++ oneDNN fused  (halo 타일링)    ── 진짜 fusion
   └─ C++ naive fused   (halo 타일링)    ── 라이브러리 호출 오버헤드 제거 시도
```

블록 유형:

| Type | 구조 | 개수 |
|---|---|---|
| A | expand 없음 (depthwise → project) | 1 |
| B | expand → depthwise → project, residual 없음 (stride 2 또는 채널 변화) | 6 |
| C | expand → depthwise → project → **add** (residual) | 10 |

## 4. 정확성 검증

모든 구현을 랜덤 입력 5개 seed로 원본 ORT 실행 결과와 비교했습니다(`rtol=1e-3, atol=1e-4`).
PyOp 단계에서는 최종 출력뿐 아니라 **블록 경계 텐서 18개 전부**를 비교했습니다.

| 구현 | 최대 오차 | 결과 |
|---|---|---|
| PyOp (fused) | ~4e-6 | 통과 (블록 경계 18개 포함) |
| C++ naive (fused, 타일링 없음) | ~6e-6 | 통과 |
| C++ oneDNN unfused | ~4e-6 | 통과 |
| C++ oneDNN fused, halo 타일링 | ~4e-6 | 통과 (budget 2.5MB, 256KB) |
| C++ naive fused, halo 타일링 | ~6e-6 | 통과 (budget 2.5MB, 256KB) |

검증하면서 찾아서 고친 버그:

- **expand 이후 경계 0 패딩 누락**: 원본 depthwise는 *expand 출력*을 0으로 패딩합니다. 그런데 타일 구현은 *block input*을 먼저 패딩한 뒤 expand를 돌려서,
  경계에 `expand(0) = bias ≠ 0`이 섞였습니다. expand 이후 진짜 이미지 경계 밖 위치를 다시 0으로 덮어써서 해결했습니다.
- **타일 경계의 가짜 0 패딩**: conv primitive에 pad=1을 넘기면 타일 경계마다 이웃 타일의 실제 데이터 대신 0이 들어갑니다.
  conv는 pad=0으로 호출하고, 패딩은 진짜 이미지 경계에서만 직접 처리하도록 바꿨습니다.
- **패치 크기 계산 순서**: 출력 타일 T를 먼저 정하면 stride 2 블록의 버퍼가 목표 budget의 약 3.5배가 됩니다.
  halo를 포함한 패치 P를 budget으로부터 먼저 구하고 `T = (P-3)/stride + 1`로 역산하도록 바꿨습니다.
- **가중치 reorder 캐시 범위**: MobileNetV2에는 shape은 같고 가중치 값은 다른 블록이 있어서, shape을 키로 하는 전역 캐시를 쓰면
  다른 블록의 가중치를 재사용하게 됩니다. 캐시를 커널 인스턴스별로 두었습니다.

## 5. 성능 결과 (wall-clock)

| 구성 (224×224, n=50) | 평균 (ms) |
|---|---|
| 원본 ORT (MLAS) | ~2 |
| C++ naive fused (타일링 없음) | ~387 |
| C++ oneDNN unfused | ~46–57 |
| C++ oneDNN fused, 타일링 사실상 없음 (budget 64MB) | 48.7 |
| C++ oneDNN fused, budget 4MB | 56.1 |
| C++ oneDNN fused, budget 1MB | 74.7 |
| C++ oneDNN fused, budget 256KB | 117–121 |

**타일을 작게 쪼갤수록 느려졌고, 어떤 budget에서도 타일링 없는 기준선보다 빨라지지 않았습니다.**
naive 타일링 구현도 같은 경향을 보였습니다. 측정 상세와 타일 크기 스윕 전체는
[docs/backend_correctness_verification.md](docs/backend_correctness_verification.md)에 있습니다.

### 입력을 키워 L3를 넘긴 실험 (224 vs 448)

"L3(12MB)가 중간 버퍼를 흡수해서 효과가 안 보인 것"이라면, 입력을 448×448로 키워 가장 큰 중간 버퍼(4.8MB → 약 19MB)가
L3를 넘게 했을 때 fused/unfused 비율이 개선돼야 합니다. 이 판정 기준은 실험 전에 고정했습니다
([npu_compiler/experiment_cache_regime.py](npu_compiler/experiment_cache_regime.py)).

판정 지표는 `R = fused / unfused`(oneDNN끼리 비교)이고, 개선률은 `(R224 − R448) / R224`입니다.
개선률이 20% 이상이면서 3회 반복 모두 같은 방향이어야 "지지"로 판정합니다.
대조군으로 fused-untiled(타일링 없이 fusion 구조만 있는 경우)를 같은 방식으로 측정했습니다.

| 스레드 | budget | R224 | R448 | 개선률 (반복별) |
|---|---|---|---|---|
| 기본 | L2 2.5MB | 1.34 | 5.55 | −313% (−256, −313, −47) |
| 기본 | L3 12MB | 1.55 | 1.03 | **+34%** (+34, +43, +30) |
| 기본 | untiled (대조군) | 1.05 | 0.76 | +28% (+50, −100, +28) |
| 1 | L2 2.5MB | 1.20 | 1.28 | −7% (−2, −11, +3) |
| 1 | L3 12MB | 1.08 | 1.40 | −30% (−30, −30, −21) |
| 1 | untiled (대조군) | 1.08 | 1.08 | +0% (0, −10, +5) |

기본 스레드의 L3 12MB 행만 보면 판정 기준을 통과합니다. 그런데 **타일링이 없는 대조군도 비슷하게 개선**됐고(반복 사이 편차도 큼),
스레드 효과를 뺀 1스레드 실험에서는 같은 조건이 오히려 **30% 악화**됐습니다. 그래서 이 개선은 타일링에 의한 메모리 효과가 아니라
멀티스레드 스케줄링 차이로 보는 것이 맞습니다. **L3를 넘겨도 타일링 효과는 관측되지 않았습니다.**

## 6. 왜 빨라지지 않았나

1. **이득의 상한이 작았습니다.** 분석적 추정으로 fusion이 없애는 중간 텐서 트래픽은 224 입력 기준 약 92MB입니다.
   이 값도 Conv와 Clip 사이 텐서까지 따로 센, 실제보다 많이 잡은 값입니다.
   메모리 대역폭을 실효 약 100GB/s로 가정하면 약 0.9ms이고, oneDNN 실행 시간(약 50ms)의 **약 2%** 입니다.
   입력을 448로 키우면 트래픽도 4배가 되지만 연산 시간도 같이 4배가 되어서, 이 비율은 거의 변하지 않습니다.
2. **캐시가 중간 버퍼를 흡수했습니다.** 224 입력에서 가장 큰 중간 버퍼는 4.8MB로, L3(12MB) 안에 들어갑니다.
   fusion을 하지 않아도 상당 부분이 DRAM까지 가지 않았습니다.
3. **타일링 비용이 이득보다 컸습니다.** 타일마다 oneDNN primitive를 실행하고 스레드를 동기화해야 합니다(naive 버전은 버퍼 초기화).
   또 halo 영역은 중복으로 계산됩니다. oneDNN은 텐서 전체를 한 번에 처리하도록 최적화돼 있어서, 타일 단위로 호출하면 그 이점을 잃습니다.
4. **커널이 느려서 workload가 compute-bound였습니다.** oneDNN의 depthwise conv는 MLAS보다 약 20배 느립니다.
   연산이 전체 시간을 지배하니 메모리에서 아낀 시간이 묻혔습니다. 커널이 MLAS 수준(약 2ms)이었다면 약 1ms의 메모리 시간은 전체의 절반쯤이 됩니다.

측정 한계: Windows VBS/Hyper-V가 켜져 있어서 VTune `memory-access`(하드웨어 이벤트 샘플링)를 쓸 수 없었습니다.
그래서 DRAM 트래픽과 캐시 미스는 실측하지 못했고, wall-clock만 측정했습니다.

## 7. 회고

- **이득만 계산하고 비용과 상한은 계산하지 않았습니다.** 분석 모델은 바이트 수만 셌고, 그 바이트가 시간으로 얼마인지,
  캐시가 얼마나 흡수하는지, 타일링 비용은 얼마인지를 넣지 않았습니다. 6장의 2% 상한은 C++ 구현 전에 계산할 수 있었던 숫자입니다.
  다음에는 roofline 분석으로 memory-bound인지부터 확인하고 시작하겠습니다.
- **fusion의 단위와 구현 방식은 분리해서 봐야 합니다.** 블록 전체를 하나로 묶고 범용 라이브러리를 타일마다 호출하는 방식은 비용이 컸습니다.
  실제 타일 추론 구현(MCUNetV2/TinyEngine 등)은 세 conv를 하나의 루프 안에서 처리하는 전용 커널을 쓰고,
  patch 기반 실행도 해상도가 높은 앞쪽 단계에만 선택적으로 적용합니다.
- **멘토가 처음부터 짚은 질문이 결국 핵심이었습니다.** 미들엔드를 끝냈을 때 멘토 피드백이
  "하나의 custom op로 묶는 것과 실제로 중간 데이터의 메모리 이동을 줄이는 것은 구분해서 검증해야 한다", 그리고
  "CPU가 정확성 검증용 환경인지, CPU 추론 성능 자체가 목표인지 구분하라"였습니다. 그래프 재작성(묶기)은 금방 끝났지만,
  메모리 이동이 실제로 줄었는지는 끝까지 직접 측정하지 못했습니다. 측정 수단(하드웨어 카운터)을 먼저 확보했어야 했습니다.
- **같은 라이브러리끼리 비교하도록 변인을 통제한 것은 효과가 있었습니다.** MLAS와 직접 비교했다면 구현 품질 차이(약 20배)와
  fusion 효과가 섞여 아무것도 결론 내릴 수 없었을 것입니다.

## 8. 프로젝트를 여기서 멈춘 이유와 다음 단계

이 프로젝트는 **위 결론에서 의도적으로 멈췄습니다.**

실험이 남긴 가장 큰 질문은 "fusion 효과를 보려면 fused 커널을 실제로 어떻게 만들어야 하는가"였습니다.
6장에서 본 것처럼, 범용 라이브러리(oneDNN)를 타일마다 호출하는 방식으로는 이득보다 비용이 컸고,
MCUNetV2/TinyEngine 같은 실제 구현은 세 conv를 하나의 루프로 처리하는 전용 커널과 정적 메모리 계획을 씁니다.
이걸 제대로 하려면 **커널 생성과 메모리 계획, 즉 컴파일러 백엔드 쪽 기초**가 먼저 필요하다고 판단했습니다.

그래서 여기서 타일 크기를 더 바꿔 보거나 커널을 손으로 계속 튜닝하는 대신, 방향을 바꿨습니다.

- **ExecuTorch 백엔드 구조 학습**: Backend Delegate Interface(논문 4.4절)를 기준으로, 실제 NPU 백엔드가 그래프 partition,
  fusion, 메모리 계획, 커널 lowering을 어떻게 나눠 처리하는지 분석
- **검증 타깃 변경**: 다시 실험한다면 메모리 시간이 전체에서 큰 비중을 차지하는 타깃(작은 on-chip 메모리의 NPU/MCU)에서,
  하드웨어 카운터로 DRAM 트래픽을 직접 잴 수 있는 환경(Linux perf 등)을 먼저 갖추고 시작

---

## 레포 구조

```
npu_compiler/            그래프 분석·재작성 + 검증/벤치마크 스크립트
  export_model.py          MobileNetV2 → ONNX export
  explore_graph.py         그래프 구조 탐색 (depthwise 판별)
  graph_utils.py           producer/consumer/value_info 맵, tensor 크기
  patterns.py              inverted residual 패턴 매칭, fused 노드 생성
  estimate.py              분석적 DRAM 트래픽 추정
  verify_pyop.py           PyOp 기반 정확성 검증 (블록 경계 포함)
  prepare_cpp_model.py     C++ 커스텀 op용 fused 모델 생성
  verify_cpp*.py           C++ 구현별 정확성 검증
  benchmark*.py            구현별 wall-clock 비교
  experiment_cache_regime.py  224 vs 448 캐시 regime 실험
experiments/             보조 진단 스크립트 (타일 스윕, VTune 대상, 스레드/구현 선택 확인, 단일 블록 디버그)
cpp_backend/             C++ 커스텀 op (OrtCustomOp API)
  inverted_residual_op.cpp              naive (타일링 없음)
  inverted_residual_op_dnnl_unfused.cpp oneDNN, 단계별 전체 버퍼
  inverted_residual_op_dnnl_fused.cpp   oneDNN, halo 타일링
  inverted_residual_op_naive_tiled.cpp  naive, halo 타일링
docs/                    계획, 검증·실험 기록
```

## 실행 방법

환경: Windows 11, Python 3.12 + [uv](https://github.com/astral-sh/uv), MSYS2 UCRT64 `g++`, oneDNN(`mingw-w64-ucrt-x86_64-onednn`),
ONNX Runtime 1.30.0 C/C++ 배포판(`third_party/onnxruntime-win-x64-1.30.0`, gitignore 처리).

```bash
uv sync
uv run python -m npu_compiler.export_model        # models/mobilenetv2.onnx
uv run python -m npu_compiler.explore_graph
uv run python -m npu_compiler.verify_pyop         # PyOp 정확성 검증
uv run python -m npu_compiler.prepare_cpp_model   # models/mobilenetv2_cpp.onnx
```

C++ 커스텀 op 빌드 (MSYS2 UCRT64 셸):

```bash
ORT=third_party/onnxruntime-win-x64-1.30.0
g++ -std=c++17 -O2 -shared -I$ORT/include cpp_backend/inverted_residual_op_naive_tiled.cpp \
    -L$ORT/lib -lonnxruntime -o cpp_backend/inverted_residual_op_naive_tiled.dll
# oneDNN 버전은 -ldnnl 추가
```

검증과 벤치마크:

```bash
uv run python -m npu_compiler.verify_cpp_dnnl_fused
INVRES_BUDGET_BYTES=262144 uv run python -m npu_compiler.verify_cpp_dnnl_fused   # 타일 budget 지정 (기본 2.5MB)
uv run python -m npu_compiler.benchmark_dnnl_fused
uv run python -m npu_compiler.experiment_cache_regime    # 224 vs 448 (models/_test_cpp_448.onnx 필요)
uv run python experiments/check_threads.py              # 보조 진단 스크립트는 레포 루트에서 경로로 실행
```

참고: `verify_cpp*.py`와 `benchmark*.py`는 Windows 전용입니다. System32에 있는 구버전 `onnxruntime.dll`과 충돌하지 않도록
venv의 `onnxruntime.dll`을 `ctypes`로 먼저 로드하는데, 이 경로가 Windows venv 구조(`.venv/Lib/...`) 기준으로 되어 있습니다.

## 문서

- [docs/PLAN.md](docs/PLAN.md): 초기 계획과 마일스톤
- [docs/backend_correctness_verification.md](docs/backend_correctness_verification.md): 정확성 검증, 성능 측정, 타일 스윕, 결론 전체 기록
- [docs/ai-collaboration-method.md](docs/ai-collaboration-method.md): AI와 협업한 방식 (핵심 로직은 직접 구현하고 AI는 리뷰와 힌트 역할)
