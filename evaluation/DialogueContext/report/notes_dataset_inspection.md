# 기존 데이터셋 조사 메모 (2026-09-25, 조사 에이전트 결과)

| 데이터셋 (경로) | 규모 / 언어 | 대화·화자 | 앞 턴 5개 이상 목표 | 사람 ko→en 정답 | 문맥 모호 태그 | 판정 |
|---|---|---|---|---|---|---|
| DialogueMT `evaluation/DialogueMT/data/dialogues.jsonl` (`bench/runs/dialogue-ko-en-text30`) | 185쌍, 30대화, ko↔en | 예, A/B | 문장 기준 35, 턴 기준 22 | 예 | 있으나 문맥용으로 심지 않음 | 보조용 |
| FLEURS `bench/runs/fleurs-ko-en-text30`, `~/datasets/fleurs` | ko, en | 아니오 (group=id) | 0 | 예 | 없음 | 부적합 (위키 문어체 독립 문장) |
| DailyTalk `evaluation/DailyTalk/` | test 1008발화/104대화 | 예 | test 488 | 아니오 (영어 원문, 한국어는 Google 번역) | 없음 | 부적합 (방향 반대, 기계 정답) |
| ACL6060 `~/datasets/acl6060` | en→10개 언어 | 독백 강연 | - | 한국어 없음 | 없음 | 부적합 |
| CoVoST2/FLEURS n-way `evaluation/ast/manifests` | de/en/ja/zh | 아니오 | - | 아니오 | 없음 | 부적합 |
| LangSwitch `evaluation/LangSwitch/` | FLEURS ko/en 이어붙인 9개 | 아니오 | - | FLEURS 경유 | 없음 | 부적합 |
| LibriSpeech, KsponSpeech, AMI | ASR 전용 | AMI 만 다화자 | - | 아니오 | 없음 | 부적합 |

근거:
- DialogueMT: 대화 25편이 6문장, 5편이 7문장. 앞 문장 5개 이상인 35문장 중 `pronoun_context` 2, `ellipsis` 6 뿐.
  태그 어휘 42종(question 42, ellipsis 36, honorific 26 …)은 문장이 무엇을 시험하는지 붙인 것이지 문맥 함정으로 설계된 것이 아니다.
- bench 의 `context_scope: group` 은 같은 group 의 앞 원문·**모델 자신의** 앞 번역을 문맥으로 넘긴다
  (`core/utils/metrics/retranslate.py::retranslate_rows`). 화자 정보는 `meta` 에만 있고 번역기로 가지 않는다.
  정답 앞 번역을 넣거나 화자 조건을 두려면 새 배관이 필요하다 — 이번 실험이 bench retranslate 를 그대로 쓰지 않은 이유.
