---
name: stity-experiment
description: Run a STiTy bench experiment end to end — pick or write pipeline/dataset configs in configs/, explain them, confirm, then run make bench or make bench-batch in tmux and watch it. Use when someone wants to run, compare or re-run an experiment, add or change a config in configs/, run several runs at once, or add a run to a batch. 실험 돌리기, 벤치 실행, 설정 만들기, 여러 실행 한 번에, 배치에 추가. Not for the live server.
---

# stity-experiment

사용자가 실험을 돌리도록 돕는다. 무엇을 돌릴지는 **짧은 질문 묶음 여러 개**로 차례로 좁혀 가고,
설정을 준비해 설명한 뒤, 마지막 질문 묶음으로 확인을 받고 실행한다.

## 지킬 것

- **사용자에게 하는 말은 전부 한국어로 쓴다.** 설정 이름·명령·파일 경로·플래그는 코드에 있는 그대로 둔다.
- **git 은 읽기만 한다.** 최근 설정을 고를 때 `git status`·`git log` 를 읽는 것 말고는 커밋·스테이징·되돌리기 모두
  하지 않고, 하라고 권하지도 않는다.
- 질문은 `AskUserQuestion` 으로 묻는다. 파이프라인만 채팅으로 말을 받는다. 한 묶음 안의 질문은 한 번에 묻고, 답을 받기 전에는 다음으로 가지 않는다.
  앞의 답으로 정해지는 질문은 건너뛴다.
- 이름 규칙은 [configs/README.md](../../../configs/README.md) 가 기준이다. 설정을 쓰거나 이름을 짓기 전에 그 파일을 읽는다.
  요약과 부품 버전 규칙은 [references/configs.md](references/configs.md).
- 저장소 루트에서 작업한다.

## 흐름

| 단계 | 하는 일 | 자세한 것 |
|---|---|---|
| 1 | 실행 형태를 묻는다 — 한 번 / 배치 / 배치에 추가 | [questions.md](references/questions.md) "1. 실행 형태" |
| 2 | 하고 싶은 실험을 말로 받아, 맞는 파이프라인을 찾거나 무엇을 만들지 알린다 | [questions.md](references/questions.md) "2. 파이프라인" |
| 3 | 데이터셋을 고르거나 단계별로 만든다 — 코퍼스 → 언어 → 범위·조건 → 효과 값 | [questions.md](references/questions.md) "3. 데이터셋" |
| 4 | 설정을 준비하고 설명한다 (묻지 않는다) | 아래 |
| 5 | 확인과 실행 방식을 묻는다 | [questions.md](references/questions.md) "5. 확인" |
| 6 | 실행한다 | [run.md](references/run.md), 배치는 [batch.md](references/batch.md) |
| 7 | 지켜보고 보고한다 | [run.md](references/run.md) "지켜보기", "결과 보고" |

## 4단계 — 설정을 준비하고 설명한다

1. **같은 내용의 설정이 있는지 먼저 찾는다.** 단계별로 고른 데이터셋이나 새 파이프라인과 내용(`meta` 빼고)이 같은
   파일이 `configs/` 에 있으면 새로 만들지 않고 그것을 쓴다고 알린다.
2. **없으면 쓴다.** 무엇을 만들거나 고쳤는지 사용자에게 알린다. 이름·버전은 [references/configs.md](references/configs.md) 를 따른다.
   규칙에 없는 경우는 이름을 하나 제안하고 설명에 "규칙에 없는 이름 — 확인 필요" 라고 적는다. 5단계에서 고칠 수 있다.
3. **실행마다 설명한다.** [references/configs.md](references/configs.md) 의 "설명하는 법" 대로 쓴다. 결과 폴더에 이미
   실행이 있으면 **덮어쓴다고 반드시 경고한다.** 파이프라인의 ASR 언어와 데이터셋 원문 언어가 다르면 그것도 적는다.
4. 스모크 테스트용 설정이 없으면 5단계 답에 따라 만들 것이라고 적어 둔다 ([references/run.md](references/run.md)).

## 6·7단계 — 실행하고 보고한다

- 스모크 테스트를 골랐으면 먼저 돌리고, 실패하면 본 실행을 시작하지 않고 원인을 보고한다.
- **지켜봄** 이면 로그를 따라가며 진행(`[ITEM] i/n`)과 오류를 알린다.
- **넘김** 이면 tmux 붙는 명령, 로그 경로, 끝났는지 알 수 있는 표시 파일을 알려 주고 끝낸다.
- **배치에 추가** 면 목록 파일에 줄을 더한 것으로 끝나고, 그 배치가 언제쯤 이 줄에 닿을지 알린다.
- 끝나면 `summary.json` 의 주요 지표, 오류·빈 출력 수, `make replay RUN=<데이터셋>/<파이프라인>` 명령을 보고한다.
