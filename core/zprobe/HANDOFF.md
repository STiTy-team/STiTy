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

## 파일

| 파일 | 역할 |
|---|---|
| `probe_z.py` | 전부 들어 있는 탐침 도구. `extract`(층별 벡터) `analyze`(검색·probe·logit lens) `tagswap`(태그만 바꿔 번역, 순도·chrF) `tssum` `steer`(평균 차/LDA 벡터 더하기, 위치·층 선택) `neurons-extract/steer` `segswap`(문장 k개 이어붙여 SEG 개수) `segprobe`(SEG 결정 층 probe, `--self`) `seglabel`(SEG 모델 의사 라벨). 환경변수 `PROBE_MODEL`(모델 경로), `PROBE_ADAPTER`(LoRA, 병합해서 로드) |
| `build_jsonl.py` | FLEURS → SFT jsonl (쌍당 N문장, 홀드아웃 지정). 태그 = 목표 언어 |
| `build_concat.py` | 문장 2개 이어붙인 음성 + `t1 <SEG> t2` 목표 (문장 경계 SEG 학습용) |
| `segtrans.py` | 분절별 번역 라벨, Hy-MT2-1.8B (말투 지시 무시함) |
| `segtrans_interp.py` | 분절별 번역 라벨, gemma-3-4b-it 통역사 프롬프트(앞 조각 문맥만) + 말투(반말/합니다체/해요체) |
| `register.py` | 한국어 어미로 말투 분포 집계 |
| `comet_eval.py` | COMET(wmt22-da) 집계. **`.venv/bin/python`** 로 실행 (comet 은 .venv 에만 있음) |
| `segreward.py` + `segreward_score.py` | stage A: 영어 경계마다 목표별 prefix 번역 → COMET 손실로 유효 판정 |
| `comet_server.py` | stage B 가 쓰는 COMET HTTP 서버 (.venv, :8777) |
| `grpo_seg.py` | stage B: SEG 모델 분절 정책 GRPO (보상 = COMET − λ·지연) |
| `distill_seg.py` | stage C: 학습된 정책으로 분절 번역 목표 생성 → SFT |
| `../../Qwen3-ASR/finetuning/qwen3_asr_sft.py` | 기존 SFT 에 `--no_seg 1`(임베딩·lm_head 고정) `--lora_targets` `final/` 저장 추가 |

학습 레시피: SEG 모델 또는 기본 모델 위에 LoRA r=64 q/k/v/o, batch 2 × grad_acc 8, 1 에폭. 소규모(≤5k)는 lr 1e-4,
FLEURS 전체는 lr 3e-5(1e-4 는 0.4 에폭부터 과적합). batch 4 는 긴 발화에서 OOM.

## 저장소 밖 산출물 (`models/zprobe/`, git 무시 대상)

| 경로 | 내용 |
|---|---|
| `models/zprobe/adapters/<이름>/` | LoRA 어댑터(peft 형식, `PROBE_ADAPTER=` 로 로드). `smoke`(13쌍 300문장) `B` `C` `D`(최소 쌍 실험) `big2`(FLEURS 전체 lr 3e-5, **번역용 최선**) `big_v1_ckpt600` `segft`·`segft2`(SEG 모델 위, 영어 ASR 만 SEG) `segft5`(+문장 경계 SEG) `segft9`(+en→ko 절 단위 SEG, Hy-MT2) `segft10_banmal`·`segft10_hapnida`(gemma 통역사 라벨, 말투) |
| `models/zprobe/data/` | 학습·검증 jsonl(음성 절대경로 포함), 평가 출력 jsonl(`ev_*`, `sg_*`, `st_*`, `srA_*`), `four.npz`(4언어 층별 벡터), `neur.npz` |
| `models/zprobe/logs/` | 학습 로그, analyze·segprobe 표, COMET 점수 |

원본은 `~/probe_z_2026-10-03/` 에도 그대로 있음. **Hugging Face(private)** 에도 올려 둠: `Doo12/stity-zprobe-adapters`(= `models/zprobe/`), `Doo12/Qwen3-ASR-1.7B-en-seg-c200`(= SEG 모델). 받기: `huggingface-cli download Doo12/stity-zprobe-adapters --local-dir models/zprobe` (읽기 토큰 필요). 올리기: `core/zprobe/hf_upload.py` (`.env` 의 `HF_TOKEN_W`). FLEURS train/dev/test 4언어 음성은 `~/datasets/fleurs/data/`.
실행 체인 쉘(`*.sh`)은 세션 scratchpad 경로가 박혀 있어 그대로는 못 씀 — 명령은 LOG.md 와 아래 참고.

## 이어서 할 일 (2026-10-04 20:00 기준)

- **stage B 대기 중.** 다른 작업의 vLLM 이 GPU 22GB 를 잡고 있어 스모크가 못 떴음. 흐름:
  1. `.venv/bin/python core/zprobe/comet_server.py --port 8777` (tmux)
  2. 스모크: `PYTHONPATH=Qwen3-ASR python core/zprobe/grpo_seg.py --seg_model models/Qwen3-ASR-1.7B-en-covost2-dailytalk-mix-c200-merged --trans_adapter models/zprobe/adapters/segft9 --utts 4 --G 4 --steps 2 --batch_utts 1 --out <out>`
  3. 본 실행: `--tgt ko --utts 200 --G 8 --steps 100 --batch_utts 2 --lam 0.02 --max_seg 6`. 로그 한 줄 = 한 스텝(meanR, meanC, mean_nseg).
     보상 해킹 감시: nseg 가 1 로 수렴하면 λ 낮추거나 `--pen` 올림.
  4. stage C: `distill_seg.py --policy_adapter <out>/adapter_step100 --trans_adapter ... --tgt ko --limit 300` → SFT → `probe_z.py segswap/tagswap` + COMET 로 seg9 와 비교.
- 스트리밍 조건 검증 아직 안 함: `segswap` 은 전체 음성을 넣고 잼. prefix 음성만 넣었을 때 같은 자리에서 끊는지 재야 함.
- ja·zh 목표 분절 라벨 만들어 4언어 SEG 어댑터 하나로 합치기 (언어당 15분).
- 품질 상한은 FLEURS 문장 수. CoVoST2 로 넘어가려면 en 음성 수십 GB 다운로드.

## 함정 (반복 금지)

- probe 는 토큰 정체를 잡기 쉬움: 마침표/쉼표 차이, `<SEG>` 앞 공백 토큰, 음성 위치 probe 는 위치만 봄. `segprobe --self` 가 최종판.
- `tagswap` 순도 지표는 `<SEG>` 글자(S·E·G)를 라틴으로 셈 — SEG 포함 출력은 순도 수치 무시.
- chrF 는 참조 말투에 묶임(반말 어댑터가 부당하게 낮음). 말투 비교는 COMET.
- `probe_z.read_rows` 는 test.tsv 고정. train 분할은 각 스크립트가 따로 읽음.
- sacrebleu 에 `ja` 토크나이저 없음(mecab) → `char`.
- 공유 GPU: 다른 프로젝트 vLLM 이 수시로 17~22GB 잡음. 체인은 여유 메모리 확인 루프 뒤에 띄울 것.
