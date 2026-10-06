# PyOp 기반 fused 모델 correctness 검증 결과

목적: fused 커스텀 op(`Inverted_residual`)이 원본(unfused) 그래프와
수치적으로 동일한 결과를 계산하는지 검증. **속도 측정 아님** —
PyOp은 Python 콜백 오버헤드가 있어 성능 지표로는 못 씀.

| seed | 최종 출력 최대 차이 | 최종 출력 allclose | 블록 경계 전체 통과 |
|---|---|---|---|
| 0 | 4.05e-06 | True | True |
| 1 | 4.11e-06 | True | True |
| 2 | 4.41e-06 | True | True |
| 3 | 4.11e-06 | True | True |
| 4 | 3.58e-06 | True | True |

**5개 시드 전체 결과: 모두 통과**

- 검증한 블록 경계 텐서 수: 18개 (17개 블록의 입력/출력, 중복 제거)
- 허용 오차: rtol=1e-3, atol=1e-4 (float32 연산 순서 차이에 의한 반올림 오차 수준)

## C++ 커스텀 op(실제 실행) 검증 결과

PyOp이 아니라 진짜 컴파일된 C++ 커스텀 op(`cpp_backend/inverted_residual_op.dll`)로 실행한 결과.

| seed | 최대 차이 | allclose |
|---|---|---|
| 0 | 5.72e-06 | True |
| 1 | 4.41e-06 | True |
| 2 | 4.83e-06 | True |
| 3 | 5.60e-06 | True |
| 4 | 3.28e-06 | True |

**전체: 모두 통과**

## 1차 성능 측정 (wall-clock, 참고용)

MLAS(원본, 최적화됨) vs 우리 naive C++ 커널(fused) 비교. 공정한 비교 아님(구현 품질 차이 포함) - 참고용 1차 수치.

- 원본(unfused, MLAS): 3.505 ms (±4.062), n=50
- fused(우리 C++, naive): 387.296 ms (±13.562), n=50
- 배율: 110.49x

## unfused + oneDNN 베이스라인 (변인 통제용)

목적: fusion 효과를 깨끗하게 분리해서 증명하려면, "같은 conv 연산 품질(oneDNN)
에서 fusion 여부만 다른" 비교가 필요함. `cpp_backend/inverted_residual_op_dnnl_unfused.cpp` -
naive 3중 for문 대신 oneDNN(`mingw-w64-ucrt-x86_64-onednn`, MSYS2 패키지)의
`convolution_forward` 프리미티브로 각 conv를 계산하되, 타일링(fusion)은
아직 적용 안 하고 각 단계(expand/depthwise/project) 결과를 기존과 동일하게
전체 크기 버퍼로 materialize.

**정확성 검증**: 5 seed, 기존과 동일한 방식으로 원본과 allclose 비교.

| seed | 최대 차이 | allclose |
|---|---|---|
| 0 | 3.58e-06 | True |
| 1 | 3.28e-06 | True |
| 2 | 2.98e-06 | True |
| 3 | 4.29e-06 | True |
| 4 | 4.05e-06 | True |

**전체: 모두 통과**

**성능 측정 (wall-clock, n=50)**:

| 구성 | 평균(ms) | 원본 대비 |
|---|---|---|
| 원본 (unfused, MLAS) | 2.1~4.1 | 1.00x |
| fused, naive C++ | 407~408 | ~100~190x |
| unfused, oneDNN | 66~75 | ~18~31x |

**구현 중 발견한 중요한 디테일**:
- oneDNN primitive(특히 JIT 커널) 생성 비용이 커서, conv 호출마다 매번
  새로 만들면 안 되고 "모양(shape)별로 1회 생성 후 재사용" 캐시가 필요함
  (`ConvShapeKey` 기반 캐시로 구현).
- `dnnl::algorithm::convolution_direct`로 강제 지정했을 때보다
  `convolution_auto`(oneDNN이 내부적으로 최적 구현을 선택, 1x1 conv는
  내부적으로 GEMM 경로를 탈 가능성)로 바꿨을 때 **약 3.7배** 빨라짐
  (unfused+oneDNN: naive 캐시 적용 후에도 ~280ms → auto 적용 후 ~70ms).

**해석**: naive 대비 oneDNN이 약 5~6배 빠르지만, 아직 fusion(타일링)이
전혀 적용 안 된 상태라 원본 MLAS 대비로는 여전히 18~31배 느림. 이는
예상된 결과 - SIMD/스레딩(oneDNN이 해결)과 메모리 트래픽(타일링이 해결)은
서로 다른 축이라, 이 숫자만으로는 fusion 효과를 아직 논할 수 없음.
다음 단계(fused+oneDNN, 타일링 적용)와 이 수치를 비교해야 fusion 자체의
순수 효과가 드러남.

### 추가 최적화 (버퍼 재사용 + 가중치 최적 레이아웃)

위 결과에서 MLAS 대비 격차가 꽤 커서(18~31배), fusion과 무관한 우리
하네스 자체의 비효율을 추가로 점검함:

1. **출력 버퍼 재사용**: 기존엔 conv 호출마다 `std::vector`를 매번 새로
   할당(`assign`)했음 - 원본 MLAS 경로는 ORT 메모리 플래너가 버퍼를
   1회만 할당해 재사용하는 것과 대조적. 커널 인스턴스(블록 하나)에
   귀속된 버퍼를 1회 할당 후 재사용하도록 변경.
2. **가중치 레이아웃을 oneDNN이 선택하게 함**: 기존엔 ONNX 원본 레이아웃
   (OIHW/GOIHW)을 강제 지정했는데, oneDNN은 보통 자체 선호 블록 레이아웃이
   있어서 이를 강제하면 성능이 덜 나옴. `format_tag::any`로 선언해 oneDNN이
   레이아웃을 고르게 하고, 가중치 값은 추론 내내 불변이므로 최적 레이아웃
   으로의 변환(reorder)을 커널 인스턴스당 1회만 수행 후 캐시.

   (주의: 이 reorder 결과 캐시는 전역이 아니라 커널 인스턴스별로 둬야 함 -
   MobileNetV2에는 shape은 같지만 가중치 값은 다른 반복 블록이 있어서,
   shape 기준 전역 캐시를 썼다면 다른 블록의 가중치를 잘못 재사용하는
   버그가 될 뻔함.)

**정확성**: 5 seed 재검증, 변화 없음(동일 오차, 모두 통과).

**성능 (wall-clock, n=50, 2회 측정)**:

| 구성 | 평균(ms) | 원본 대비 |
|---|---|---|
| 원본 (unfused, MLAS) | ~1.9~2.1 | 1.00x |
| fused, naive C++ | ~387~388 | ~187~201x |
| unfused, oneDNN (버퍼 재사용 + 최적 레이아웃) | **~51~57** | **~26~27x** |

1차(66~75ms, 18~31x)에서 추가로 약 20~25% 더 빨라짐. 여전히 MLAS 대비
20배 이상 느리지만 격차가 꾸준히 줄고 있음 - 남은 격차는 주로 MLAS의
MobileNet 특화 튜닝(특히 depthwise conv)과의 차이로 추정되며, fusion
(타일링) 전 단계에서 fair한 "같은 라이브러리" 베이스라인으로 쓰기에는
충분히 안정된 상태.
