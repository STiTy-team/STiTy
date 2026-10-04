# zprobe — 공통 의미 표현(z) 탐침과 언어 공통 번역 어댑터 (2026-10-03 ~ 10-04)

얼린 Qwen3-ASR-1.7B 안에 "무슨 말을 했는지" 가 언어와 무관하게 담긴 자리(z)가 있는지 찾고, 그걸 바탕으로
**모델 하나·태그 하나로 어느 언어 쌍이든 번역**하게 만드는 스파이크. 상세 수치·표는 전부 [LOG.md](LOG.md)
(실험 순서대로 쌓은 기록). 여기는 결론과 이어서 할 일만.

## 결론 (각 항목의 근거 표는 LOG.md 같은 제목 아래)

1. **z 있음.** en/ko/ja/zh 음성 4언어, LLM L7~L15 에 공통 내용 표현(검색 피크 L9, 합동 갤러리 0.89~0.99).
   인코더 출력엔 없음. L19 부터 토큰 철자 단계로 넘어가며 사라짐. 언어 판별 방향은 전 층 공존.
2. **z 는 영어 쪽에 놓임.** 평균 차 벡터 한 번 더하면 X→en 은 0.9~1.0 전환, en→X 는 절반. 비영어 목표 조립은
   학습이 필요. `<asr_text>` 위치 L24 가 en 소스에 최선 레버(0.80), 음성 위치 L15 가 X→en 에 최선.
3. **미세조정은 쌍별 데이터 없이 됨.** FLEURS 쌍당 300문장 LoRA(6분)로 안 본 쌍(zh→ko/ja→ko/zh→ja) 0.92~0.98,
   COMET 0.83~0.89, ASR 퇴행 없음. 한 방향 3쌍(ASR + en→X 또는 X→en)만으로도 전 방향 열림. 목표 언어 조립
   능력은 언어 간 공유(ko 목표 데이터 0 으로도 en→ko 0.92). FLEURS 전체(31.6k)로 늘려도 동일 — 문장 다양성 한계,
   다음 스케일링은 CoVoST2(ko 없음, en→ja/zh 289k).
4. **SEG(분절).** SEG 미세조정은 z 를 안 바꿈. 경계 정보는 z 에 있음(쉼표에서 끊을지 probe AUC L9 0.90, L24 0.97),
   토큰은 L28 에서 방출. 번역 출력으로 SEG 습관은 제로샷 전이 안 됨. 목표 언어마다 절 단위 SEG 예시가 필요하고
   (소스는 하나면 됨: en→ko 로 가르치면 zh→ko 에도 나옴), 라벨은 자동(SEG 모델 분절 + 로컬 MT 분절별 번역).
   말투(반말/합니다체)도 라벨대로 목표 언어 조립에 붙어 소스 무관 전이.
5. **언어별 분절 정책 필요.** 영어 경계를 목표별 prefix 번역+COMET 으로 채점하면 유효 ko 0.65 / ja 0.59 / zh 0.65 이고
   언어 간 상관 0.13~0.34 — 60% 가 언어별로 유불리 갈림. → GRPO 로 목표 언어별 분절 정책 학습(stage B) 착수.
6. **1차 GRPO(λ=0.02)는 경계를 고르지 않고 더 잘게 잘랐다.** koen-seg-mix 베이스, ko 목표, 100 스텝: 분절 수 2.5→3.2,
   COMET 0.83→0.80, 보상은 평평. 증류해 어댑터로 만들면 en-ko COMET(SEG 제거 채점) ft9mix 0.887 / 대조군(자체 분절+prefix 번역
   라벨) 0.859 / 정책 0.846. 격차가 라벨 COMET 격차(0.887/0.870/0.843)와 같다 — **병목은 라벨 생성기(prefix 번역)**, 그 다음이
   정책의 과분절. zh-ko 는 셋 다 0.86 으로 같음(소스 간 전이 유지). 증류 경로 자체는 끝까지 동작.

## 파일

| 파일 | 역할 |
|---|---|
| `probe_z.py` | 전부 들어 있는 탐침 도구. `extract`(층별 벡터) `analyze`(검색·probe·logit lens) `tagswap`(태그만 바꿔 번역, 순도·chrF) `tssum` `steer`(평균 차/LDA 벡터 더하기, 위치·층 선택) `neurons-extract/steer` `segswap`(문장 k개 이어붙여 SEG 개수) `segprobe`(SEG 결정 층 probe, `--self`) `seglabel`(SEG 모델 의사 라벨). 환경변수 `PROBE_MODEL`(모델 경로), `PROBE_ADAPTER`(LoRA, 병합해서 로드) |
| `build_jsonl.py` | FLEURS → SFT jsonl (쌍당 N문장, 홀드아웃 지정). 태그 = 목표 언어 |
| `build_concat.py` | 문장 2개 이어붙인 음성 + `t1 <SEG> t2` 목표 (문장 경계 SEG 학습용) |
| `segtrans.py` | 분절별 번역 라벨, Hy-MT2-1.8B (말투 지시 무시함) |
| `segtrans_interp.py` | 분절별 번역 라벨, gemma-3-4b-it 통역사 프롬프트(앞 조각 문맥만) + 말투(반말/합니다체/해요체) |
| `register.py` | 한국어 어미로 말투 분포 집계 |
| `comet_eval.py` | COMET(wmt22-da) 집계. `--strip_seg` 를 주면 `<SEG>` 를 빼고 채점 — 원문 채점은 SEG 많은 출력을 따로 벌준다(같은 쌍에서 −0.11 vs −0.04). LOG 의 기존 수치는 원문 채점. comet 패키지가 있는 venv 로 실행 — 어느 venv 에 있는지는 머신마다 다르다 (skkai 는 `.venv-tpm`). 학습·탐침 쪽 venv(peft·qwen_asr) 와 다른 venv 다 |
| `segreward.py` + `segreward_score.py` | stage A: 영어 경계마다 목표별 prefix 번역 → COMET 손실로 유효 판정 |
| `comet_server.py` | stage B 가 쓰는 COMET HTTP 서버 (comet 이 있는 venv, :8777) |
| `grpo_seg.py` | stage B: SEG 모델 분절 정책 GRPO (보상 = COMET − λ·지연) |
| `distill_seg.py` | stage C: 학습된 정책으로 분절 번역 목표 생성. `--ids_from <train jsonl>` 로 seg9 와 같은 발화만, `--policy_adapter none` 이면 SEG 모델 자체 분절(대조군) |
| `build_distill_train.py` | 증류 행으로 seg9 식 train jsonl 의 en-<tgt> 행을 교체. 음성 경로를 이 머신 FLEURS 로 고치고, 행마다 `audio/text/pair` 만 남기고(필드가 섞이면 `datasets` 로더가 죽는다), 반복 루프 행을 버린다 |
| `../../Qwen3-ASR/finetuning/qwen3_asr_sft.py` | 기존 SFT 에 `--no_seg 1`(임베딩·lm_head 고정) `--lora_targets` `final/` 저장 추가 |

학습 레시피: SEG 모델 또는 기본 모델 위에 LoRA r=64 q/k/v/o, batch 2 × grad_acc 8, 1 에폭. 소규모(≤5k)는 lr 1e-4,
FLEURS 전체는 lr 3e-5(1e-4 는 0.4 에폭부터 과적합). batch 4 는 긴 발화에서 OOM.

## 저장소 밖 산출물 (`models/zprobe/`, git 무시 대상)

| 경로 | 내용 |
|---|---|
| `models/zprobe/adapters/<이름>/` | LoRA 어댑터(peft 형식, `PROBE_ADAPTER=` 로 로드). `smoke`(13쌍 300문장) `B` `C` `D`(최소 쌍 실험) `big2`(FLEURS 전체 lr 3e-5, **번역용 최선**) `big_v1_ckpt600` `segft`·`segft2`(SEG 모델 위, 영어 ASR 만 SEG) `segft5`(+문장 경계 SEG) `segft9`(+en→ko 절 단위 SEG, Hy-MT2) `segft10_banmal`·`segft10_hapnida`(gemma 통역사 라벨, 말투) |
| `models/zprobe/data/` | 학습·검증 jsonl(음성 절대경로 포함), 평가 출력 jsonl(`ev_*`, `sg_*`, `st_*`, `srA_*`), `four.npz`(4언어 층별 벡터), `neur.npz` |
| `models/zprobe/logs/` | 학습 로그, analyze·segprobe 표, COMET 점수 |

원본은 만든 머신의 `~/probe_z_2026-10-03/` 에 있고, **Hugging Face(private)** 에도 올려 둠 — 머신 간 이동은 이쪽으로:
`Doo12/stity-zprobe-adapters` → `models/zprobe/`, `Doo12/Qwen3-ASR-1.7B-en-seg-c200` → `models/Qwen3-ASR-1.7B-en-covost2-dailytalk-mix-c200-merged/`
(SEG 모델, 4.4GB). 받기: `.env` 의 `HF_TOKEN`(Doo12) 으로 `snapshot_download(repo, local_dir=...)` 또는 `huggingface-cli download <repo> --local-dir <dir>`.
올리기: `core/zprobe/hf_upload.py` (`.env` 의 `HF_TOKEN_W`).
FLEURS train/dev/test 4언어 음성은 `~/datasets/fleurs/data/`. stage B·C 는 영어 음성만 쓰지만 목표 언어의
`train.tsv` 텍스트는 필요하다 — 없으면 `google/fleurs` 데이터셋 레포의 `data/<lang>/train.tsv` 만 받아 넣는다.
2026-10-04 이전 실행 체인 쉘(`*.sh`)은 세션 scratchpad 경로가 박혀 있어 그대로는 못 씀 — 명령은 LOG.md 와 아래 참고. koen-mix 체인은 `chains/`.

## 이어서 할 일 (2026-10-04 22:30 기준)

베이스는 `models/Qwen3-ASR-1.7B-koen-seg-mix-merged`(HF 캐시 `Doo12/Qwen3-ASR-1.7B-koen-seg-mix-merged` 심볼릭 링크).
그 위의 번역 어댑터는 `models/zprobe/adapters/segft9_mix`, 체인 쉘은 `chains/mix_chain.sh`·`chains/mix_control.sh`
(경로가 이 머신 기준이라 그대로 실행 가능). 실행 순서는 쉘에 있고 수치는 LOG.md 마지막 절.

- **라벨 생성기부터 고친다** (정책보다 먼저). `grpo_seg.trans_gen` 과 `distill_seg.gen` 의 prefix 번역에 반복 억제가
  없어 루프가 난다(증류 행 7/282, 출력 1/60). `repetition_penalty` 또는 `no_repeat_ngram_size` 를 넣고, 조각마다 붙는
  마침표를 라벨에서 정리할지 정한다. 이 번역기가 stage B 보상도 만들므로 보상 잡음도 같이 준다.
- **보상식.** λ=0.02 는 "더 잘게" 가 "더 잘" 과 같은 값이 되게 한다. 후보: λ≤0.01, 또는 발화별 무분절 COMET 을 빼서
  상대 보상으로(지연으로 품질 손실을 못 사게). 정책 평가는 증류까지 가지 말고 stage A 방식(경계별 COMET 손실)으로
  먼저 재는 편이 싸다.
- koen-mix 베이스로 최종 어댑터를 만들 땐 **ko-ko 행도 `seglabel --pair ko-ko`** 로 의사 라벨 — 평문 ko-ko 행이 한국어 SEG 를
  지운다(ko-ko ASR SEG 2.0 → 0.33).
- 스트리밍 조건 검증 아직 안 함: `segswap` 은 전체 음성을 넣고 잼. prefix 음성만 넣었을 때 같은 자리에서 끊는지 재야 함.
- ja·zh 목표 분절 라벨 만들어 4언어 SEG 어댑터 하나로 합치기 (언어당 15분). 이 머신엔 zh·ja test 음성을 받아 두었다.
- 품질 상한은 FLEURS 문장 수. CoVoST2 로 넘어가려면 en 음성 수십 GB 다운로드.

## 함정 (반복 금지)

- probe 는 토큰 정체를 잡기 쉬움: 마침표/쉼표 차이, `<SEG>` 앞 공백 토큰, 음성 위치 probe 는 위치만 봄. `segprobe --self` 가 최종판.
- `tagswap` 순도 지표는 `<SEG>` 글자(S·E·G)를 라틴으로 셈 — SEG 포함 출력은 순도 수치 무시.
- chrF 는 참조 말투에 묶임(반말 어댑터가 부당하게 낮음). 말투 비교는 COMET.
- `probe_z.read_rows` 는 test.tsv 고정. train 분할은 각 스크립트가 따로 읽음.
- sacrebleu 에 `ja` 토크나이저 없음(mecab) → `char`.
- 공유 GPU: 다른 프로젝트 vLLM 이 수시로 17~22GB 잡음. 체인은 여유 메모리 확인 루프 뒤에 띄울 것.
- 받은 data jsonl 의 음성 경로는 만든 머신 홈이 박혀 있다. 다른 머신에선 `sed` 로 홈을 바꾸거나 `build_distill_train.py` 를 거친다.
- FLEURS 는 언어·분할마다 따로 받는다. 없는 분할의 음성을 `segswap` 이 만나면 그 쌍만 비어서 넘어가므로 로그에서 `FileNotFoundError` 를 확인할 것.
