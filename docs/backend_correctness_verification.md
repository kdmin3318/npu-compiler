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
