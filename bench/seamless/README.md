# bench/seamless/ — SeamlessStreaming 을 bench 와 같은 조건으로 재기

SeamlessStreaming 은 번역까지 모델 하나가 한다(speech-to-speech). ASR → 번역으로 이어지는 bench 파이프라인에
들어가지 않으므로 여기서 따로 돌린다. 대신 **입력 오디오와 채점 코드는 bench 것을 그대로 쓴다.**

```bash
tmux new-session -d -s seamless -c <repo> "bash bench/seamless/run_seamless.sh > logs/chain_seamless.log 2>&1"
```

## 단계

| 단계 | 스크립트 | env | 하는 일 |
|---|---|---|---|
| 1 | `export_items.py` | bench | `fleurs_{en-ko,ko-en}_vol23` 을 bench 와 똑같이 불러 음량을 맞춘 뒤 wav 와 manifest 로 쓴다 |
| 2 | `seamless_run.py --task s2s` | `asr-seamless` | 200ms 조각을 실시간 속도로 넣어 번역 음성을 합성한다. 번역 글이 나올 때마다 그때까지 넣은 오디오 위치를 남긴다 |
| 3 | `seamless_run.py --task asr` | `asr-seamless` | 같은 모델을 목표 언어 = 원래 언어로 돌려 원래 언어 전사를 낸다 |
| 4 | `seamless_score.py` → `bench.metrics.comet` | bench, comet | 합성 음성을 Qwen3-ASR 로 되받아쓰고 bench 의 함수로 채점한다 |

`run_seamless.sh` 가 넷을 차례로 돌린다. 2·3 은 언어마다 10문장을 먼저 돌려 절반 넘게 실패하거나 비면
전체를 건너뛴다. 결과는 `work/`(git 밖)에 쌓이고, 채점 요약은 `results/<pair>.json` 으로 옮겨 둔다.

엔진(`engine_seamless.py`)은 이 브랜치에 없다. 4차에 SeamlessStreaming 을 붙인 `feat/omni-seamless-backends`
작업 트리의 `evaluation/backends/` 에 있고, `SEAMLESS_BACKENDS` 로 그 경로를 준다. env 를 만드는 법도 거기
(`setup_seamless_clean.sh`)에 있다 — `fairseq2==0.2.1` 이 `torch==2.2.2` 전용이라 순서가 중요하다.

## 지표가 무엇을 재는가

| 지표 | 어디서 | 캐스케이드와 다른 점 |
|---|---|---|
| WER·CER | 3단계 전사 모드 | S2S 실행이 원문을 얼마나 알아들었는지가 아니다. S2S 경로는 원문 전사를 밖으로 내지 않는다. 같은 모델에 전사만 시킨 값이다 |
| ASR-COMET | 2단계 합성 음성 → `Qwen/Qwen3-ASR-1.7B` 되받아쓰기 → bench COMET | 한 번 더 받아쓰므로 그 ASR 의 오류가 섞인다. 캐스케이드 COMET 과 완전히 같은 척도는 아니다 |
| YAAL | 2단계 번역 글이 나온 시점의 오디오 위치 | bench 와 같은 계산(omnisteval, computation-unaware) |
| Token Emission | — | 원문 단어가 화면에 뜨는 시각이 없어서 내지 않는다 |

## 알려진 결함

**한국어로 내보내면 무너진다.** 전사 모드의 한국어 전사 268개 중 196개 앞에 "MBC 뉴스 김민형입니다"가
붙고(CER 0.775), en→ko 번역 음성을 되받아쓴 글 268개 중 256개에 "MBC 뉴스"나 "인터뷰"가 끼어 있다
(ASR-COMET 0.486). 같은 한국어 음성을 영어로 번역시키면 정상 영어가 나온다(ASR-COMET 0.714). 알아듣지
못하는 게 아니라 한국어로 써내지 못하는 체크포인트 결함이다. 쓸 만한 것은 ko→en S2S 하나다.

ko→en 에서 `ko_1814`, `ko_1914`, `ko_2005` 세 문장은 번역 음성이 합성되지 않았다(`no audio chunks produced`).
