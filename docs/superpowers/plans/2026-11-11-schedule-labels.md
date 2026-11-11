# prefix14 방출 일정표 라벨 — 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 번역 어댑터의 학습 라벨을 "발화당 방출 일정표"로 바꾸고, 두 타깃 10,000 발화 규모의 새 코퍼스(prefix14)로 SFT→GRPO 학습 체인을 GRPO(p13t) 종료 뒤 tmux 에 건다.

**Architecture:** 순수 함수 모듈(`schedule_gate.py`: 토큰화·일정표 검사)을 먼저 만들고 라벨러 CLI(`label_schedule.py`)와 행 생성기(`build_prefix14.py`)가 그것을 쓴다. 입력은 소스별로 모은 문장 매니페스트(`sentences.jsonl`, 같은 원문은 녹음 목록으로 묶음), 출력은 `train_prefix14.jsonl`/`val_prefix14.jsonl`. 라벨 위생 지표와 벤치용 지어냄 지표는 별도 스크립트. 체인 스크립트가 전부를 잇는다.

**Tech Stack:** Python 3.10 (conda `stity` = `.venv`), openai 클라이언트(Letsur 게이트웨이, `LETSUR_API_KEY`), Qwen3ForcedAligner(`align_more.py`), Hy-MT2(`core/translator/local_translator.HyMTTranslator`), cometkiwi(`.venv-tpm`), pytest(`.venv/bin/python -m pip install pytest` 1회).

**Spec:** `docs/superpowers/specs/2026-11-11-schedule-labels-design.md`

## Global Constraints

- 작업 트리 `/home/skkai/STiTy-wt/p13`, 브랜치 `exp/schedule-labels`. 새 코드는 `core/research/zprobe/labels/`, `core/research/zprobe/eval/`, `core/research/zprobe/train/` 아래.
- 비용 발생 호출은 호출마다 `<out>.usage.jsonl` 에 append, 로그에 누적 비용 출력, `--budget` 상한(1차 luna 20). 크래시 재실행 시 usage 로그를 지우지 않는다.
- 몇 분 이상 도는 것은 tmux. 환경변수는 명령 문자열 안에서 `.env` 를 읽는다.
- `stall_flush_sec` 는 쓰지 않는다. 런타임 코드는 바꾸지 않는다.
- 밀림 상한 N: 영어 원문 8단어, 한국어 원문 6어절. 홀드아웃 2%(문장 단위). 두 타깃 목표 10,000 발화.
- 학습: LoRA r64/α128, lr 1e-4, 1 epoch, batch 2×8, `--lazy_features 1`, `--load_best 1`, `--no_seg 1`. GRPO: B 레시피, `--pairs en-ko ko-en --source_tag 0 --steps 400`.
- 벤치는 매니저 queue 로만. 설정의 모델 경로는 절대경로. GRPO 와 벤치는 GPU 를 같이 쓰지 않는다.
- 공용 `STiTy-team/datasets/kosp2e/` 는 건드리지 않는다(별도 폴더에서 받는다).

## Review Focus

1. 라벨러가 단어 번호를 1 이 아니라 0 부터 세거나 원문 토큰화와 다르게 셀 때 — `after_word` 가 범위를 벗어나거나 한 칸씩 밀린 일정표는 게이트에서 "range"로 거부되어야 한다 (Task 1 테스트).
2. 한국어 원문의 어절과 정렬기 토큰이 다를 때(공백 무시 병합) — 어절 k 의 끝 시각이 토큰 누적 영숫자 수로 정확히 잡혀야 한다 (Task 1 `eojeol_end_times` 테스트).
3. 일정표의 마지막 항목이 비어 있거나 전체가 빈 경우 — 라벨러가 "wait" 만 내면 완결성 게이트가 거부해야 한다 (Task 1 테스트).
4. 녹음이 여러 개인 문장(CoVoST 중복)에서 홀드아웃이 녹음 단위로 갈리면 누수 — 같은 `sid` 의 녹음은 모두 같은 쪽에 가야 한다 (Task 6 테스트).
5. 벤치 조각이 정답 분절 밖(무음·다음 분절)에서 나올 때 지어냄 지표가 분모를 잘못 세는 경우 — 원문이 없는 시점의 조각은 "위반"이 아니라 "판정 불가"로 따로 센다 (Task 8 테스트).

---

### Task 1: 일정표 토큰화·검사 모듈 `schedule_gate.py`

**Files:**
- Create: `core/research/zprobe/labels/schedule_gate.py`
- Test: `core/research/zprobe/labels/test_schedule_gate.py`

**Interfaces:**
- Consumes: `build_prefix.content(text, tgt) -> set[str]` (내용어 집합, 기존).
- Produces:
  - `words(src: str, lang: str) -> list[str]` — 공백 분리 토큰(ko 어절, en 단어). 구두점은 토큰에 붙어 있다.
  - `clause_ends(tokens: list[str], lang: str) -> list[int]` — 절이 끝나는 토큰 번호(1-based) 목록. en: `[.!?;:,]` 로 끝나는 토큰과 `and/but/so/because/if/when/while/that/which` 바로 앞 토큰; ko: 토큰 끝이 `[.!?,]` 이거나 `고/서/면/데/지만/니까` 로 끝나는 어절. 마지막 토큰은 항상 포함.
  - `parse_schedule(obj) -> list[tuple[int, str]]` — `{"schedule":[{"after":k,"text":s},...]}` 를 `(after, text)` 목록으로. 형식이 틀리면 `ValueError("format: ...")`.
  - `check_schedule(items, tokens, lang, tgt, *, known_at: Callable[[int], set[str]], full_known: set[str], N: int, clause_end: list[int]) -> str | None` — 위반 이유 문자열 또는 None. 검사 순서: range(1 ≤ after ≤ len(tokens), 비감소), all-empty, lookahead(`content(text,tgt) - known_at(after)` 중 `full_known` 에 있는 것), clause(항목 내용어의 원문 위치 j 가 속한 절 끝 c 에 대해 after > c), deferral(after − j > N). 원문 위치 j 는 `known_at` 의 증가 지점으로 구한다: 내용어 w 가 처음 나타나는 가장 작은 k.
  - `eojeol_end_times(tokens: list[str], word_times: list[dict]) -> list[float]` — 토큰 k 의 끝 시각. `word_times` 는 `[{"w":..,"end":..}]`(정렬기 토큰). `build_prefix13._span_bounds` 와 같은 누적 영숫자 수 매핑.

- [ ] **Step 1: 실패하는 테스트 작성** — `test_schedule_gate.py`

```python
def test_words_ko_eojeol():
    assert words("회의가 끝나면 보고서를 보낼게요.", "ko") == ["회의가", "끝나면", "보고서를", "보낼게요."]

def test_parse_schedule_rejects_missing_after():
    with pytest.raises(ValueError): parse_schedule({"schedule": [{"text": "x"}]})

def test_check_range_and_order():
    toks = words("The manager approved the budget", "en")
    assert check_schedule([(0, "매니저가")], toks, "en", "ko", known_at=lambda k: set(), full_known=set(), N=8, clause_end=[5]).startswith("range")
    assert check_schedule([(3, "a"), (2, "b")], toks, "en", "ko", known_at=lambda k: set(), full_known=set(), N=8, clause_end=[5]).startswith("range")

def test_check_all_empty():
    toks = words("Hello there", "en")
    assert check_schedule([(2, "")], toks, "en", "ko", known_at=lambda k: set(), full_known=set(), N=8, clause_end=[2]) == "all additions are empty"

def test_check_lookahead():
    toks = words("I will send the report tomorrow", "en")
    known = {1: set(), 2: set(), 3: {"보내"}, 4: {"보내"}, 5: {"보내", "보고"}, 6: {"보내", "보고", "내일"}}
    why = check_schedule([(3, "내일 보낼게요")], toks, "en", "ko", known_at=lambda k: known[k], full_known={"보내", "보고", "내일"}, N=8, clause_end=[6])
    assert why.startswith("lookahead") and "내일" in why

def test_check_deferral_cap():
    toks = words("w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12", "en")
    known = lambda k: {"매니"} if k >= 1 else set()
    assert check_schedule([(12, "매니저가")], toks, "en", "ko", known_at=known, full_known={"매니"}, N=8, clause_end=[12]).startswith("deferral")
    assert check_schedule([(9, "매니저가")], toks, "en", "ko", known_at=known, full_known={"매니"}, N=8, clause_end=[12]) is None

def test_check_clause_end():
    toks = words("He left, and she stayed.", "en")  # clause ends at 2 ("left,") and 5
    known = lambda k: {"떠났"} if k >= 2 else set()
    assert check_schedule([(5, "그는 떠났고")], toks, "en", "ko", known_at=known, full_known={"떠났"}, N=8, clause_end=clause_ends(toks, "en")).startswith("clause")

def test_eojeol_end_times_merged_tokens():
    toks = ["자각몽을", "꾸었다"]
    wt = [{"w": "자각", "end": 0.4}, {"w": "몽을", "end": 0.8}, {"w": "꾸었다", "end": 1.3}]
    assert eojeol_end_times(toks, wt) == [0.8, 1.3]
```

- [ ] **Step 2: 실패 확인** — Run: `cd /home/skkai/STiTy-wt/p13 && .venv/bin/python -m pytest core/research/zprobe/labels/test_schedule_gate.py -v` — Expected: ImportError/FAIL.
- [ ] **Step 3: `schedule_gate.py` 구현** — 위 시그니처대로. 모듈 상단에서 `sys.path` 에 `labels/` 를 넣고 `from build_prefix import content`. 이유 문자열은 접두사로 종류를 밝힌다: `"range: ..."`, `"all additions are empty"`, `"lookahead: 내일 - ..."`, `"clause: ..."`, `"deferral: ..."`.
- [ ] **Step 4: 통과 확인** — 같은 명령, Expected: 8 passed.
- [ ] **Step 5: Commit** — `git add core/research/zprobe/labels/schedule_gate.py core/research/zprobe/labels/test_schedule_gate.py && git commit -m "feat(zprobe): schedule_gate - 일정표 라벨 토큰화·검사"`

### Task 2: 문장 매니페스트 `build_prefix14_manifest.py`

**Files:**
- Create: `core/research/zprobe/labels/build_prefix14_manifest.py`
- Test: `core/research/zprobe/labels/test_prefix14_manifest.py`

**Interfaces:**
- Consumes: `models/zprobe/data/prefix13/aligned.jsonl`(기존 FLEURS·CoVoST 녹음: `audio, lang, tgt, src, sid`), FLEURS `train.tsv`, CoVoST `train.tsv`(`/home/skkai/datasets/covost2_train_en/`), AI Hub `datasets/aihub71693/ko/manifest.jsonl`(`audio, offset, duration, reference.transcript, reference.translations.en, group`), kosp2e zeroth 분할 xlsx + Zeroth train manifest(`reference.transcript`), `--extra-covost 2000`.
- Produces: `models/zprobe/data/prefix14/sentences.jsonl` 한 줄 = `{"sid": str, "lang": "en"|"ko", "tgt": "ko"|"en", "src": str, "ref": str|null, "source": "fleurs"|"covost"|"aihub"|"kosp2e", "recs": [{"audio": "<abs path or path#start=..#end=..>"}]}`. 같은 `(lang, src)` 는 한 줄로 합치고 `recs` 에 녹음을 모은다. `sid` = `f"{source}_{lang}_{hash8(src)}"`. 함수 `dedup(rows) -> list[dict]`, `select_aihub(manifest_rows, n_talks, domains) -> list[dict]`(도메인 EA/MA/NA/SA 균등, 문장 수 많은 발표부터), `select_covost_extra(tsv_rows, used_wavs, n, rng) -> list[dict]`.

- [ ] **Step 1: 실패하는 테스트** — `dedup` 이 같은 src 두 녹음을 한 줄 `recs` 2개로 합치고 `sid` 가 같은지; `select_aihub` 이 4도메인에서 고르게 뽑는지(발표 8개 → 도메인당 2); CoVoST 추가 선택이 `used_wavs` 를 피하는지.
- [ ] **Step 2: 실패 확인** — `pytest core/research/zprobe/labels/test_prefix14_manifest.py -v`.
- [ ] **Step 3: 구현** — CLI 인자: `--aligned`, `--fleurs-root ~/datasets/fleurs/data`, `--covost-root /home/skkai/datasets/covost2_train_en`, `--extra-covost 2000`, `--aihub-manifest`, `--aihub-sentences 5000`, `--kosp2e-dir <별도 폴더>`, `--kosp2e-xlsx`, `--zeroth-manifest`, `--kosp2e-sentences 3000`, `--out`, `--seed 14`. AI Hub 문장의 `audio` 는 `f"{abs}#start={offset:.3f}#end={offset+duration:.3f}"`. 끝에 방향·출처별 문장 수와 녹음 수를 출력한다.
- [ ] **Step 4: 통과 확인**.
- [ ] **Step 5: Commit** — `feat(zprobe): prefix14 문장 매니페스트 - 소스 모으기·중복 묶기`

### Task 3: 데이터 받기 `fetch_prefix14_data.sh`

**Files:**
- Create: `core/research/zprobe/labels/fetch_prefix14_data.sh`

**Interfaces:**
- Produces: AI Hub ko 발표 약 45개(`datasets/aihub71693/ko/manifest.jsonl` 갱신), `models/zprobe/data/prefix14/kosp2e/`(별도 폴더: `data/zeroth_train.xlsx` 3,600행 + `data/zeroth/*.wav`), Zeroth train manifest(`datasets/zeroth` 에 `SPLIT=train`).

- [ ] **Step 1: 스크립트 작성** — 순서: (1) `set -a; . /home/skkai/STiTy/.env; set +a; export AIHUB_APIKEY="$AIHUB_API_KEY"`; (2) AI Hub: 라벨 zip 목록에서 ko 발표 id 를 도메인별 문장 수로 정렬해 45개 고르고 `LANG_SRC=ko TALKS=... ./aihub71693/install.sh`; (3) `SPLIT=train ./zeroth/install.sh`; (4) kosp2e: 상류 `split/zeroth_train.xlsx` 를 받아 Zeroth train manifest 에 전사가 있는 id 중 3,600행만 남긴 xlsx 를 `models/zprobe/data/prefix14/kosp2e/data/` 에 쓰고, `install.sh` 의 다운로드 부분만(convert 제외) 그 폴더를 작업 폴더로 실행. 마커 `P14D MARK/FAIL/DONE`.
- [ ] **Step 2: tmux 로 실행** — `tmux new-session -d -s p14d -c /home/skkai/STiTy-wt/p13 "bash core/research/zprobe/labels/fetch_prefix14_data.sh 2>&1 | tee /tmp/p14d.log"`.
- [ ] **Step 3: 확인** — `grep P14D_DONE /tmp/p14d.log`; AI Hub ko manifest 문장 수 ≥ 5,500; kosp2e wav 수 ≥ 3,400; zeroth train manifest 22,263행.
- [ ] **Step 4: Commit** — `feat(zprobe): prefix14 데이터 받기 스크립트`

### Task 4: 새 녹음 강제 정렬 (`align_more.py --manifest`)

**Files:**
- Modify: `core/research/zprobe/lid/align_more.py` (recs 수집부 36~41행 뒤에 `--manifest`·`--out` 인자)

**Interfaces:**
- Consumes: Task 2 의 `sentences.jsonl`.
- Produces: `models/zprobe/data/prefix14/word_times.jsonl` — 한 줄 `{"id": sid+"#"+n, "audio": <rec audio>, "lang", "words": [{"w","end"}]}`. 끝 시각은 녹음 자체 기준(`#start` 가 있으면 그 구간을 잘라 정렬하고 `end` 는 구간 시작 기준 상대 시각).

- [ ] **Step 1: 인자 추가** — `--manifest <sentences.jsonl>` 가 주어지면 기존 FLEURS/conv 수집을 건너뛰고 매니페스트의 모든 `recs` 를 `add(id, audio, lang, src)` 로 넣는다. `--out` 기본값은 그대로, prefix14 는 `--out models/zprobe/data/prefix14/word_times.jsonl`. `librosa.load` 에 `#start/#end` 를 적용(`qwen3_asr_sft.load_audio` 와 같은 규칙, 함수 복사 금지 — `Qwen3-ASR/finetuning/qwen3_asr_sft.py:load_audio` 를 import).
- [ ] **Step 2: 소규모 확인** — `--manifest` 에 10문장짜리 임시 파일로 실행, 출력 10줄·`words` 비어 있지 않음.
- [ ] **Step 3: 전체 실행(tmux, GPU 4GB)** — `tmux new-session -d -s p14a -c ... "PYTHONPATH=Qwen3-ASR .venv/bin/python -u core/research/zprobe/lid/align_more.py --manifest models/zprobe/data/prefix14/sentences.jsonl --out models/zprobe/data/prefix14/word_times.jsonl 2>&1 | tee /tmp/p14a.log"`. 이미 `prefix12/word_times.jsonl` 에 있는 FLEURS·CoVoST 녹음은 재사용하도록 `--reuse models/zprobe/data/prefix12/word_times.jsonl` 로 복사해 온다.
- [ ] **Step 4: 확인** — `ALIGN_MORE_DONE ok=... fail=...`, fail 비율 < 2%.
- [ ] **Step 5: Commit** — `feat(zprobe): align_more --manifest - 임의 녹음 목록 정렬`

### Task 5: 단어 접두사 번역 `hy_word_prefixes.py` (지어냄 게이트 재료)

**Files:**
- Create: `core/research/zprobe/labels/hy_word_prefixes.py`
- Test: `core/research/zprobe/labels/test_hy_word_prefixes.py` (토큰 분할·캐시 키만, 모델 없이)

**Interfaces:**
- Consumes: `sentences.jsonl`, `HyMTTranslator`(`build_connective_en.py hyprefix` 의 로딩·배치 코드와 같은 방식).
- Produces: `models/zprobe/data/prefix14/hy_prefixes.jsonl` — 한 줄 `{"sid", "prefixes": [t1, t2, ..., tK]}`, `tk` = 원문 토큰 1..k 의 Hy 번역(타깃 언어). `known_sets(prefixes, tgt) -> list[set[str]]` 는 `content(tk, tgt)` 의 누적 합집합(k 가 늘면 집합은 줄지 않는다).

- [ ] **Step 1: 테스트** — `known_sets` 가 누적 합집합인지; 접두사 수 = 토큰 수.
- [ ] **Step 2: 구현** — 배치 64, 길이순 정렬, 출력은 `sid` 단위로 append(재개 가능). GPU 4GB.
- [ ] **Step 3: 전체 실행(tmux `p14h`)** — 약 18k 문장 × 평균 12 토큰 = ~22만 접두사. 로그에 진행률.
- [ ] **Step 4: 확인·Commit** — 줄 수 = 문장 수. `feat(zprobe): hy_word_prefixes - 단어 접두사 번역`

### Task 6: 라벨러 `label_schedule.py`

**Files:**
- Create: `core/research/zprobe/labels/label_schedule.py`
- Create: `core/research/zprobe/labels/prompts/schedule_rules.txt` (규칙 1~6 원문, 설계 3.1 그대로)
- Test: `core/research/zprobe/labels/test_label_schedule.py` (`ask` 를 가짜로 바꿔 재시도·usage·예산 로직만)

**Interfaces:**
- Consumes: Task 1 전부, Task 5 `known_sets`, `relabel_adds_en.py` 의 클라이언트·usage·예산 패턴(복사해 쓰되 공통부를 함수로 뽑지 않아도 된다).
- Produces: `models/zprobe/data/prefix14/schedule_<model>.jsonl` — 한 줄 `{"sid", "lang", "tgt", "schedule": [[after, text], ...], "attempts": n}`; `<out>.usage.jsonl`; `FAIL sid: reason` 로그; 끝에 `LABEL_SCHEDULE_DONE ok= fail= est $`. `--completeness` 는 2단계로: 참조가 있으면 Task 7 의 통계 단계에서 COMET-da, 없으면 cometkiwi — 이 스크립트 안에서는 하지 않고 `--skip-completeness` 상태로 둔다(GPU 모델을 라벨 워커와 같이 올리지 않기 위해).

- [ ] **Step 1: 테스트** — 가짜 `ask` 가 1차에 `range` 위반, 2차에 정상을 주면 `attempts==2` 로 저장; 두 번 다 실패하면 FAIL 로 빠지고 저장 안 함; `spent > budget` 이면 호출 없이 종료 코드 2.
- [ ] **Step 2: 구현** — 프롬프트: `Source sentence ({lang}), words numbered: 1:The 2:manager ...`, `Reference translation ({tgt}): ...`(없으면 줄 생략), 규칙 파일 내용, `Answer with JSON only: {"schedule": [{"after": k, "text": "..."}, ...]}`. 게이트 호출: `check_schedule(items, tokens, lang, tgt, known_at=lambda k: known[k-1], full_known=known[-1], N=8 if lang=="en" else 6, clause_end=clause_ends(tokens, lang))`. 인자: `--sentences`, `--hy`, `--model gpt-6-luna`, `--reasoning medium`, `--workers 64`, `--budget 20`, `--limit`, `--out`. 재개: `out` 에 있는 `sid` 건너뜀.
- [ ] **Step 3: 통과 확인**.
- [ ] **Step 4: 파일럿 50문장** — `--limit 50 --out models/zprobe/data/prefix14/schedule_pilot.jsonl`. 실패율 < 40% 이고 FAIL 이유 분포가 한 종류에 몰리지 않으면 진행, 아니면 프롬프트 손질 후 다시(파일럿 비용 < $0.1).
- [ ] **Step 5: 전량 실행(tmux `p14l`)** — `--out models/zprobe/data/prefix14/schedule_luna.jsonl`. 마커 `LABEL_SCHEDULE_DONE`.
- [ ] **Step 6: Commit** — `feat(zprobe): label_schedule - 방출 일정표 라벨러`

### Task 7: 완결성 게이트 + 라벨 위생 `schedule_stats.py`

**Files:**
- Create: `core/research/zprobe/labels/schedule_stats.py`
- Test: `core/research/zprobe/labels/test_schedule_stats.py`

**Interfaces:**
- Consumes: `schedule_luna.jsonl`, `sentences.jsonl`, `hy_prefixes.jsonl`(Hy 전체 번역 = 마지막 접두사), cometkiwi/COMET-da(`.venv-tpm`).
- Produces: (1) `schedule_luna.gated.jsonl` — 완결성 통과분(참조 있으면 COMET-da(joined, ref) ≥ 0.70, 없으면 kiwi(joined) ≥ kiwi(hy_full) − 0.08); (2) 표준출력 표 — 방향별 문장 수, 빈 구간 비율, 마지막 1/3 토큰에 몰아 냄 비율(= 모든 항목의 `after` 가 토큰 수의 2/3 이후), 첫 항목까지 단어 수 중앙값·p95, 항목당 밀림(`after − 첫 내용어 위치`) 중앙값·최대, 번역/원문 길이비 중앙값·p5·p95, 게이트 탈락 수. `--old models/zprobe/data/prefix13/aligned.jsonl` 을 주면 옛 코퍼스의 같은 지표(구 단위 근사)를 옆 열로.
- 순수 함수: `stats_for(rows: list[dict], tokens_fn) -> dict`.

- [ ] **Step 1: 테스트** — 토이 일정표 3개로 몰아 냄 비율·첫 항목 단어 수 계산.
- [ ] **Step 2: 구현·실행** — `.venv-tpm/bin/python core/research/zprobe/labels/schedule_stats.py --schedule ... --sentences ... --hy ... --old ... --out-gated ...` (GPU, tmux `p14s`).
- [ ] **Step 3: 판정** — en→ko 몰아 냄 ≤ 5%, 밀림 최대 ≤ N, 방향 간 빈 구간 비율 차이 2배 안. 벗어나면 Task 6 Step 4 로 돌아가 프롬프트를 고친다(사용자에게 수치와 함께 보고).
- [ ] **Step 4: Commit** — `feat(zprobe): schedule_stats - 완결성 게이트·라벨 위생 지표`

### Task 8: 행 생성 `build_prefix14.py`

**Files:**
- Create: `core/research/zprobe/labels/build_prefix14.py`
- Test: `core/research/zprobe/labels/test_build_prefix14.py`

**Interfaces:**
- Consumes: `schedule_luna.gated.jsonl`, `sentences.jsonl`, `word_times.jsonl`, `schedule_gate.eojeol_end_times`, `silence_tail_4s.wav`, prefix13 의 reset 행 규칙(`build_prefix13.py` 224~253행, 태그 없이).
- Produces: `models/zprobe/data/train_prefix14.jsonl`, `val_prefix14.jsonl` — 행 `{"audio", "cond", "text", "pair", "prompt": ""}`, `cond` 는 `"language {Target}<asr_text>" + 지금까지 낸 것`. 순수 함수:
  - `emit_times(items, end_times, pad=0.2) -> list[float]` — 항목별 방출 시각 = `end_times[after-1] + pad`.
  - `grid(dur, emit_times, step=0.5) -> list[float]` — 0.5초 격자 ∪ 방출 시각, 첫 1.0초 이후만, 오름차순 중복 제거.
  - `rows_on_time(items, end_times, dur, pair, audio) -> list[dict]` — 격자마다 조건/정답(설계 4절 규칙).
  - `rows_catchup(items, end_times, dur, pair, audio, rng, n=2, lag=(1,2)) -> list[dict]` — 조건을 `lag` 항목 뒤로.
  - `rows_end(items, pair, audio) -> list[dict]` — 발화 끝(잘림 없음)에 남은 전부 1행 + 문장 통째 행 1행(`cond` 접두사만, `text` = 전체).
  - `rows_silence(...)`, `rows_seg_pairs(...)`, `rows_reset(...)` — 기존 규칙 이식.
  - `split_holdout(sids: list[str], frac=0.02, seed=14) -> set[str]`.

- [ ] **Step 1: 테스트** — (a) 토이 일정표 `[(2,"매니저가"),(9,"새 사무실 예산을"),(10,"어제 승인했어요")]`, `end_times` 10개, dur 5.0 → `rows_on_time` 에서 2.5초 전 행은 `text==""`, "매니저가" 방출 행의 `cond` 가 접두사만이고 `text=="매니저가"`, 마지막 행 `cond` 에 앞 두 항목이 들어 있다; (b) `rows_catchup` 의 `text` 가 밀린 항목을 공백으로 이어 붙였다; (c) `split_holdout` 이 같은 `sid` 의 녹음을 가르지 않는다(녹음 단위 입력이 아니라 sid 단위라는 것을 테스트로 고정); (d) 같은 문장의 녹음 2개가 각자 `word_times` 로 다른 시각의 행을 만든다.
- [ ] **Step 2: 실패 확인 → 구현** — 인자: `--schedule`, `--sentences`, `--word-times`, `--sil-wav`, `--out`, `--val-out`, `--rows-per-rec 8`(정시 행 상한, 넘치면 무작위 추출), `--catchup-per-rec 2`, `--sil-cap 800`, `--seg-cap 2500`, `--reset-rows 3000`, `--seed 14`. 끝에 종류별 행 수와 빈 정답 비율을 출력(`BUILD_PREFIX14_DONE`).
- [ ] **Step 3: 통과 확인·실행** — 행 수 90k~140k 범위, 빈 정답 비율 보고.
- [ ] **Step 4: Commit** — `feat(zprobe): build_prefix14 - 일정표에서 학습·검증 행 생성`

### Task 9: 벤치 지어냄 지표 `lookahead_rate.py`

**Files:**
- Create: `core/research/zprobe/eval/lookahead_rate.py`
- Test: `core/research/zprobe/eval/test_lookahead_rate.py`

**Interfaces:**
- Consumes: 벤치 런 폴더(`items.jsonl`: `records[*].{type, target_lang, translation, decision_audio_sec}`, `reference_segmentation[*].{offset, duration, src_lang, transcript}`, `reference_alignment[*].{word, start, end}`), Hy 번역(런 뒤 후처리, GPU).
- Produces: 표준출력 `LOOKAHEAD run=<name> pieces=<n> judged=<m> violations=<v> rate=<v/m> unjudged=<n-m>`; 순수 함수 `heard_words(alignment, t, lag=0.0) -> list[str]`, `judge(piece_content: set, known: set, full: set) -> bool`(위반이면 True), 조각이 속한 분절을 `decision_audio_sec` 로 찾고 분절 밖이면 unjudged.

- [ ] **Step 1: 테스트** — 토이 정렬 3단어, t=1.0 에 2단어만 들림; `judge({"내일"}, known={"보고"}, full={"보고","내일"})` → True; 분절 밖 조각은 unjudged.
- [ ] **Step 2: 구현** — 분절별로 들린 접두사 번역을 Hy 로 캐시(같은 접두사 재사용). prefix13tr 런에 먼저 돌려 기준선 값을 기록.
- [ ] **Step 3: Commit** — `feat(zprobe): lookahead_rate - 벤치 출력의 지어냄 비율`

### Task 10: 체인·설정·큐

**Files:**
- Create: `core/research/zprobe/train/prefix14_chain.sh`
- Create: `configs/pipelines/brig.koen+prefix14-lidreset.yml`, `brig.koen+prefix14-grpo-lidreset.yml` (gitignore; `prefix13tr-*` 에서 모델 경로만 `models/zprobe/merged/prefix14_full`, `prefix14_grpo_cerenlat_full` 로, 절대경로)

**Interfaces:**
- Consumes: Task 2~8 산출물, `prefix13tr_chain.sh` 의 SFT/GRPO/merge 블록, `p13dir_chain.sh` 의 큐 투입 블록.
- Produces: `models/zprobe/adapters/prefix14/final`, `merged/prefix14_full`, `merged/prefix14_grpo_cerenlat_full`, 매니저 job 2개. 마커 `P14 MARK/FAIL/SFT_READY/DONE`.

- [ ] **Step 1: 체인 작성** — 순서: (0) `/tmp/p13t_chain.log` 에 `P13T_EXIT` 가 찍힐 때까지 대기(2분 간격) → (1) `build_prefix14_manifest.py` → (2) Task 3 데이터 확인(없으면 FAIL) → (3) `align_more.py --manifest` → (4) `hy_word_prefixes.py` → (5) `label_schedule.py`(`--budget 20`) → (6) `schedule_stats.py`(gated 출력, 위생 표를 로그에) → (7) `build_prefix14.py` → (8) SFT(`--lazy_features 1 --load_best 1`, train/val prefix14) → merge → `P14_SFT_READY` → 매니저에 `brig.koen+prefix14-lidreset` 투입(run-now) → GRPO 는 그 벤치가 `JOB-END` 될 때까지 `roomy 20000` 로 대기 → GRPO → merge → `brig.koen+prefix14-grpo-lidreset` 투입 → `P14_DONE`. 단계마다 파일 존재로 멱등. 각 단계 끝에 `df -h /` 한 줄.
- [ ] **Step 2: 스모크** — `--limit 20` 매니페스트로 (1)(3)(4)(5)(7)을 임시 출력 경로로 한 번 돌려 형식을 확인(라벨 비용 < $0.05). SFT 는 `--max_steps 5`(스크립트에 없으면 `--epochs 0.001` 대신 행 100개 파일로) 로 1분 확인.
- [ ] **Step 3: 큐** — `tmux new-session -d -s p14 -c /home/skkai/STiTy-wt/p13 "bash core/research/zprobe/train/prefix14_chain.sh 2>&1 | tee /tmp/p14_chain.log"`. 데이터 받기(Task 3)는 이와 별개로 먼저 돌아가 있어야 한다.
- [ ] **Step 4: Commit** — `feat(zprobe): prefix14 체인 - 라벨→정렬→행→SFT→GRPO→벤치 큐`

### Task 11: 보고

- [ ] 라벨 위생 표(옛 vs 새), 비용 합계(`usage.jsonl` 합), 관문 표(ko/en COMET·LongYAAL, prefix13tr·prefix10f 나란히), 잘린 문장 수와 COMET, 온전한 문장 COMET, ko 슬롯 멈춤 횟수, 지어냄 비율(prefix13tr 기준선 vs prefix14). 설계 2절의 실패 조건 둘을 명시적으로 판정.
