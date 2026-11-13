# 핸드오프: prefix15 벤치 2개 (+ 이어서 prefix15 GRPO) 를 다른 머신에서

**할 일: prefix15 SFT 모델과 prefix14 SFT 모델을 `northstar_ko+en-ko` 로 벤치하고, ko·en 두 방향 점수를 낸다.
그다음 prefix15 GRPO 400 스텝 → 병합 → 같은 벤치.** 4090 은 다른 프로젝트가 GPU 를 오래 잡고 있어 여기서 못 돌린다.

## 상태 (2026-11-13 13:30, 4090)

| 항목 | 값 |
|---|---|
| 브랜치 | `exp/schedule-labels` (origin) |
| prefix15 라벨 | 17,808 문장 (`schedule.gated.jsonl`, 판정 통과·구제분 포함). 라벨 비용은 4090 의 usage.jsonl 에 이미 기록됨 — **라벨은 다시 돌리지 말 것** |
| prefix15 SFT | 182,645 행 / val 1,500, LoRA r64 1 epoch, 2h59m, eval_loss 0.943, eval_cer 0.300 |
| prefix15 병합본 | `models/zprobe/merged/prefix15_full` → HF `Doo12/zprobe-prefix15-full` 본체 |
| prefix14 SFT 병합본 | HF `Doo12/zprobe-prefix14-full` (벤치는 한 번도 안 돎) |
| 벤치 | 둘 다 미실행. 4090 에서는 설정 yml description 의 콜론 때문에 제출 실패 → 고친 yml 이 HF extras 에 있음 |
| GRPO | 미실행 |

## 1. 받을 것 (HF 비공개, 토큰은 `.env` 의 `HF_TOKEN_W`)

| repo | 종류 | 쓰는 곳 |
|---|---|---|
| `Doo12/zprobe-prefix15-full` | model | prefix15 병합 모델 본체 + `extras/` |
| `Doo12/zprobe-prefix14-full` | model | prefix14 병합 모델 본체 (그 repo 의 `extras/` 는 prefix14 벤치 설명서 겸용) |
| `Doo12/Qwen3-ASR-1.7B-koen-seg-mix-merged` | model | 베이스 ASR (`asr.model`) |
| `Doo12/stity-prefix14-bundle` | dataset | GRPO 에만 필요: `sentences.jsonl`·`word_times.jsonl`·오디오 (README 의 §1~2 대로 배치) |

`zprobe-prefix15-full/extras/`:
- `configs/pipelines/brig.koen+prefix15-lidreset.yml`, `brig.koen+prefix15-grpo-lidreset.yml`, `brig.koen+prefix14-lidreset.yml`
- `configs/datasets/northstar_ko+en-{ko,en}.yml`
- `zprobe_data/lid_phone_profiles.json`, `zprobe_data/prefix15/schedule.gated.jsonl`
- `adapters/prefix15/final/` (SFT LoRA, 참고용 — 병합본이 있으면 안 씀)

```bash
export HF_TOKEN=<HF_TOKEN_W>
for r in zprobe-prefix15-full:models/zprobe/merged/prefix15_full zprobe-prefix14-full:models/zprobe/merged/prefix14_full \
         Qwen3-ASR-1.7B-koen-seg-mix-merged:models/Qwen3-ASR-1.7B-koen-seg-mix-merged; do
  python -c "from huggingface_hub import snapshot_download as s; print(s('Doo12/${r%%:*}', local_dir='<repo>/${r#*:}'))"
done
```

## 2. 배치

- `extras/configs/pipelines/*.yml` → `<repo>/configs/pipelines/`, `extras/configs/datasets/*.yml` → `<repo>/configs/datasets/`
  (둘 다 gitignore 라 git 에 없다).
- 파이프라인 yml 세 개의 절대경로를 이 머신에 맞게 고친다: `asr.model`, `adapter.model`, `adapter.lid_profiles`.
  4090 경로(`/home/skkai/STiTy/...`, `/home/skkai/STiTy-wt/p13/...`)가 그대로 들어 있다.
- `extras/zprobe_data/lid_phone_profiles.json` → `<repo>/models/zprobe/data/`,
  `extras/zprobe_data/prefix15/schedule.gated.jsonl` → `<repo>/models/zprobe/data/prefix15/`.
- 데이터셋: `STITY_DATA_ROOT` 아래 `northstar-260928/` (STiTy-team/datasets 클론, 또는 `zprobe-prefix14-full/extras/datasets/`).
  `.env` 에 `STITY_DATA_ROOT=<절대경로>` 가 없으면 bench·manager 가 시작하자마자 멈춘다.
- yml 을 고친 뒤 `python -c "import yaml; yaml.safe_load(open('<yml>'))"` 로 셋 다 읽히는지 확인한다.

## 3. 벤치 2개 (prefix15, prefix14)

이 머신이 매니저(`make bench-manager`)에 붙어 있으면 매니저 queue 로 제출한다(run-now). 없으면 tmux 로 하나씩, **동시에 돌리지 말 것**
(yml 의 `gpu_memory_utilization: 0.30` 은 24GB 기준, 다른 GPU 작업과 겹치면 vLLM 이 죽는다):

```bash
# 순서: prefix15 먼저, 끝나면 prefix14
tmux new-session -d -s p15bench -c <repo> "make bench CONFIG=brig.koen+prefix15-lidreset DATASET=northstar_ko+en-ko > /tmp/p15bench.log 2>&1; \
  make bench CONFIG=brig.koen+prefix14-lidreset DATASET=northstar_ko+en-ko > /tmp/p14bench.log 2>&1"
```

결과: `bench/runs/northstar_ko+en-ko@v2/<pipeline>/summary.json` 의 `metrics`.

## 4. 후처리 (ko 런에서 en 점수까지) — 파이프라인마다

```bash
P=brig.koen+prefix15-lidreset   # 그다음 brig.koen+prefix14-lidreset
uv run --project bench python scripts/brig_rescore.py --pipeline $P --src-dataset northstar_ko+en-ko --dst-dataset northstar_ko+en-en
uv run --project bench/metrics/comet python -m bench.metrics.comet --run-dir bench/runs/northstar_ko+en-en@v2/$P
for t in ko en; do PYTHONPATH=. python core/research/zprobe/eval/lookahead_rate.py --run-dir bench/runs/northstar_ko+en-$t@v2/$P --target $t --grace 2.0; done
```

보고(방향 ko / en 각각): COMET, LongYAAL(ms), `commit_reasons.lid`, lookahead 비율. 비교 기준 (4090 측정, 같은 데이터셋):

| 파이프라인 | en→ko COMET | en→ko LongYAAL | ko→en COMET | ko→en LongYAAL |
|---|---|---|---|---|
| `brig.koen+prefix13tr-lidreset` | 0.7101 | 2712 ms | 0.6196 | 1867 ms |
| `brig.koen+prefix13-grpo-lidreset` | 0.7106 | 2249 ms | 0.6440 | 2043 ms |

## 5. prefix15 GRPO (벤치 2개 끝난 뒤)

4090 의 `p15_grpo.sh` 와 같은 명령. `D14` 는 prefix14 bundle 에서 받은 `sentences.jsonl`·`word_times.jsonl` 위치.

```bash
# 빈 VRAM 20GB 확인 뒤, COMET 서버 먼저
.venv-tpm/bin/python -u core/research/zprobe/train/comet_server.py --port 8777 > models/zprobe/logs/comet_server_p15.log 2>&1 &
# "ready" 가 로그에 찍히면
PYTHONPATH=Qwen3-ASR .venv/bin/python -u core/research/zprobe/train/grpo_prefix.py \
  --policy_model models/zprobe/merged/prefix15_full --out models/zprobe/grpo/prefix15_grpo_cerenlat --source_tag 0 \
  --steps 400 --lam 0.1 --sil_frac 0.25 --mu 0.25 --anchor 1 --hard_frac 0.5 --nu 0.3 --rho 0.2 \
  --nli vicgalle/xlm-roberta-large-xnli-anli --pairs en-ko ko-en --utts 400 \
  --schedule models/zprobe/data/prefix15/schedule.gated.jsonl --sentences $D14/sentences.jsonl --word_times $D14/word_times.jsonl \
  --sigma_s 0.2 --sigma_l 0.1 --lat_floor 1 >> models/zprobe/logs/grpo_prefix15_cerenlat.log 2>&1
```

- 스크립트로 떨궈 tmux 로 띄운다. 로그에 `GRPO_DONE` 이 찍히면 마지막 `adapter_step*` 을 `models/zprobe/adapters/prefix15_grpo_cerenlat` 로 복사.
- 병합: `PYTHONPATH=Qwen3-ASR .venv/bin/python core/research/zprobe/train/merge_adapter.py --base models/zprobe/merged/prefix15_full --adapter models/zprobe/adapters/prefix15_grpo_cerenlat --out models/zprobe/merged/prefix15_grpo_cerenlat_full`
- 그다음 `brig.koen+prefix15-grpo-lidreset` 로 §3~4 반복.
- 보고: 로그의 step 줄 `R= COMET= lat= emit= wait=` 의 1·100·200·300·400 스텝 추이.

## 6. 함정

- 체인·학습은 conda `stity` 환경을 `.venv` 로 링크, COMET 은 `.venv-tpm` (+kiwipiepy). 벤치는 `uv run --project bench` 로 따로.
- 몇 분 넘게 도는 일은 tmux 로 (`.claude/rules/long-jobs-in-tmux.md`). tmux 안에서는 `.env` 를 직접 읽는다.
- 돌고 있는 bash 스크립트를 제자리 편집하지 말 것 (조각 읽기로 문법오류 사망, 11-12 실제 사고).
- yml `description` 에 따옴표 없는 콜론을 넣지 말 것 — 4090 벤치 제출이 이것 때문에 실패했다.
- 1.7B 경로는 dot commit 기본값이 켜져 있다. yml 에 `dot_commit_confirm: true` 가 명시돼 있으니 그대로 둔다.
