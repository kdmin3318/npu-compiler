# AI와 함께 코딩 실력 늘리기: 스캐폴딩 협업 방법론

## 결론부터

우리가 하려는 "스켈레톤(뼈대) + TODO" 방식은 즉흥적으로 만든 방법이 아니라, 교육학에서 검증된
**Scaffolding-and-Fading(비계 설정과 점진적 제거)** / **Faded Worked Examples(단계적으로
줄어드는 풀이 예시)** 기법과 원리가 같다. AI 시대 개발자 실력 유지에 대한 실무 서베이들도
결이 같은 원칙을 말한다.

## 이론적 근거

**Vygotsky의 근접발달영역(ZPD)**: 학습자가 혼자 할 수 있는 것과, 전문가 도움을 받아야 할 수
있는 것 사이의 공간에서 학습 효과가 가장 크다. 이 공간을 메워주는 임시 지지대가 "스캐폴딩"이고,
학습자가 숙달됨에 따라 지지대를 점점 걷어내는 게 "fading"이다.

**Faded Worked Examples**: worked example(문제 + 풀이 과정 + 정답)에서 풀이 단계를 단계적으로
제거해 나가며, 학습자가 점점 더 많은 부분을 스스로 채우게 하는 기법. 초반엔 대부분을 예시로 주고
핵심 단계만 빈칸으로 두다가, 뒤로 갈수록 빈칸(직접 채워야 하는 부분)을 늘려간다. 구조가 잘 잡힌
도메인(우리 프로젝트처럼 명확한 입출력이 있는 그래프 알고리즘)에서 특히 효과적이라고 알려져 있다.

## AI 시대 실무자들의 원칙 (여러 서베이 공통)

1. **어려운 부분일수록 먼저 직접 풀어본다** — 실력은 어려운 문제를 스스로 풀 때 는다. 먼저
   혼자 구현하고, 그 다음 AI 결과와 비교해서 놓친 부분을 확인하는 식으로 쓰면 AI가 "코드
   리뷰어" 역할을 하게 되고 본인이 직접 짠 게 유지된다.
2. **AI-free 시간을 의도적으로 둔다** — 전혀 AI 도움 없이 코딩하는 시간을 스스로 확보한다.
3. **"이건 AI한테 맡겨도 되는가"를 매번 의식적으로 판단한다** — 이 판단력 자체가 AI 시대의
   핵심 역량이라는 게 공통된 지적이다.
4. **손으로 먼저 짜고, AI는 멘토로 쓴다** — 설명/디버깅/리팩토링 제안 용도로 활용하고, 초안
   자체를 AI에 맡기지 않는다.
5. **AI가 만든 코드는 반드시 이해하고 검토한다** — boilerplate/반복 작업엔 AI를 쓰되, 그
   결과를 그대로 받아쓰지 않고 이해한 뒤 사용한다.

## 이 프로젝트에 적용하는 방식

- **핵심 로직(패턴 매칭, fusion 재작성)은 100% 직접 작성.** Claude는 스켈레톤/boilerplate
  제공, 막혔을 때 힌트, 완성된 코드 리뷰 역할만 한다.
- **스켈레톤 fading 전략**: 마일스톤이 진행될수록 제공하는 뼈대의 범위를 줄여간다.
  - 1단계(그래프 구조 파악): 모델 로딩/출력 I/O는 제공, 노드 판별 함수는 TODO
  - 2단계(패턴 매칭): 판별 함수는 이미 완성했으니, 이제 시퀀스 탐지 로직 전체가 TODO
  - 3단계(그래프 재작성): 얕은 helper만 제공, 재작성 로직은 전부 TODO
- **막혔을 때 Claude에게 묻는 방식**: "정답 코드를 줘"가 아니라 "이 접근이 맞는 방향인지",
  "왜 이 케이스에서 안 되는지"를 질문하는 방식으로 쓴다. 코드는 최후의 수단으로만 받는다.

## 참고자료

- [How to use AI to write code without skill atrophy](https://www.augmentedswe.com/p/use-ai-without-skill-atrophy)
- [Avoiding Skill Atrophy in the Age of AI](https://addyo.substack.com/p/avoiding-skill-atrophy-in-the-age)
- [AI Coding Assistants: Best Practices for Developers to Retain Core Skills](https://medium.com/@martin.jordanovski/ai-coding-assistants-best-practices-for-developers-to-retain-core-skills-012e71fa31c7)
- [Suggestions for graduated exposure to programming concepts using fading worked examples (ACM)](https://dl.acm.org/doi/abs/10.1145/1288580.1288594)
- [An exploratory study on fade-in versus fade-out scaffolding for novice programmers (Springer)](https://link.springer.com/article/10.1007/s12528-021-09307-w)
- [Modelling, Fading, and Scaffolding in Education and Learning Design](https://insights.insendi.com/read/scaffolding-modelling-and-fading-in-learning-design)
