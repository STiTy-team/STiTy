# prefix15 닫힌 단위 방출 일정표 — 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** prefix14 의 문장·정렬·Hy 접두사를 그대로 두고, 닫힌 단위 규칙(v5)으로 라벨을 다시 달아 — 형태 게이트 → COMET → LLM 결격 판정 → 재라벨 → 기다림 라벨 구제 — 새 코퍼스(prefix15)로 SFT·병합·벤치까지 4090 에서 체인 하나로 돌린다.

**Architecture:** 라벨러(`label_schedule.py`)·형태 게이트(`schedule_gate.py`)·판정(`judge_schedule.py`)·행 생성(`build_prefix14.py`)의 새 부분은 이미 커밋돼 있다(7d1488e4). 남은 것은 (1) 그 코드의 테스트, (2) 판정 탈락분 재라벨 입력(`--only-sids`/`--notes`), (3) 판정 결과 합치기 + 기다림 라벨 + 위생표 새 열(`schedule_stats.py`), (4) `auto_go.py` 기준, (5) 체인 `prefix15_chain.sh` 와 4090 핸드오프.

**Tech Stack:** Python 3.12 (`.venv`: kiwipiepy, openai, pytest), COMET (`.venv-tpm`), Letsur 게이트웨이 `gpt-6-luna`(`LETSUR_API_KEY`), bash 체인 + tmux, 벤치 매니저 API.

**Spec:** `docs/superpowers/specs/2026-11-12-prefix15-closed-unit-schedule-design.md`

## Global Constraints

- 라벨·판정 호출은 모두 `.usage.jsonl` 에 append 되고, 재실행 때 지우지 않는다(`.claude/rules/cost-watch.md`). 예산 `--budget 35`(라벨) / `--budget 10`(판정).
- 라벨 규칙 파일은 `prompts/schedule_rules_v2.txt` 그대로(v5 본문). 라벨러는 `--rules … --deferral-cap 0 --reasoning low --model gpt-6-luna`.
- 형태 게이트: 한국어 맨 명사 끝 거부(다음 조각이 잇는 조사면 통과), 영어 매달린 기능어 끝 거부(and/or/but/so 단독 허용), 앞머리 문장부호 이동, 같은 after 합침. 이미 구현됨.
- 붙이기: 앞 조각이 맨 명사로 끝나고 뒤 조각이 조사·어미로 시작할 때만 공백 없이(`glues`). 지시관형사(이/그/저/…)로 시작하는 두 어절 이상 조각은 단어.
- 판정 코드는 `grammar/order/mistranslation/omission/addition` 만, 하나라도 있으면 탈락. 참조가 있으면 프롬프트에 넣는다.
- COMET-da ≥ 0.70 / cometkiwi ≥ Hy − 0.08. 탈락·FAIL·ERROR 는 버리지 않고 기다림 라벨 `[(K, 전체 번역)]`, `rescue: "wait"`.
- 위생표 기준: en-ko dump_share ≤ 0.15, first_after_median ≤ 8, rescued_wait ≤ 0.25. 기다림 라벨은 모든 열에서 따로 센다.
- 학습 레시피·행 생성 인자는 prefix14 체인과 같다. 출력 이름은 `prefix15` (`train_prefix15.jsonl`, `adapters/prefix15`, `merged/prefix15_full`).
- 오래 도는 단계는 tmux. 돌고 있는 체인 파일은 제자리 편집하지 않는다.

## Review Focus

1. 조각이 조사 하나("도")뿐인데 앞 조각이 문장부호로 끝나는 경우 — `glues` 는 False 여야 하고 shape 검사는 통과해야 한다 (Task 1).
2. 판정 탈락 → 재라벨에서 라벨러가 또 형태 게이트에 2회 걸리면 FAIL 로 `.fail.jsonl` 에 가고, 그 sid 는 기다림 라벨이 되어야 한다 — 어느 파일에도 안 남는 sid 가 0 이어야 한다 (Task 3·4).
3. 참조 없는 CoVoST 의 기다림 라벨은 Hy 번역인데, Hy kiwi 가 하위 10% 면 코퍼스에서 빠진다 — 그 문장 수가 위생표에 찍혀야 한다 (Task 4).
4. 같은 sid 가 S1(1차 라벨)과 S2(재라벨) 둘 다에 있을 때 최종 파일엔 하나만 — S2 가 판정을 통과하면 S2, 아니면 기다림 (Task 3).
5. `judge_schedule.py` 가 JSON 을 못 받으면(`malformed`) 그 문장은 탈락이 아니라 "재판정 대상" 으로 남아야지 기다림으로 가면 안 된다 (Task 2).

---

### Task 1: 커밋된 게이트·붙이기 코드의 테스트

**Files:**
- Modify: `core/research/zprobe/labels/test_schedule_gate.py`
- Modify: `core/research/zprobe/labels/test_build_prefix14.py`
- Modify: `core/research/zprobe/labels/test_label_schedule.py`
- Modify: `core/research/zprobe/labels/schedule_gate.py` (영어 `'s` 쪼갬 거부 한 줄)

**Interfaces:**
- Consumes: `schedule_gate.shape_check(items, tgt) -> str | None`, `glues(prev, text) -> bool`, `join_texts(texts) -> str`, `ko_continues(text) -> bool`, `fix_leading_punct(items) -> items`, `merge_same_after(items, covers) -> (items, covers)`; `build_prefix14._delta(cond_items, new_items) -> str`; `label_schedule.build_prompt(row, note)`, `RULES`, `DEFERRAL_CAP`, `n_cap(lang)`.
- Produces: 없음(테스트). `shape_check` 에 영어 `'s`/`'re`/`n't` 로 시작하는 조각 거부가 추가된다: 이유 문자열 `shape: piece i "…" splits a word - join it to the previous piece`.

- [ ] **Step 1: 실패하는 테스트 쓰기** (`test_schedule_gate.py` 에 추가)

```python
def test_shape_bare_noun_rejected_unless_particle_follows():
    assert shape_check([(2, "이 문장"), (4, "특별히"), (7, "당신을 위해 만들어졌습니다.")], "ko").startswith("shape: piece 1")
    assert shape_check([(4, "응답률"), (6, "도"), (7, "더 높습니다.")], "ko") is None
    assert shape_check([(1, "그것은"), (3, "영국의"), (4, "가장 오래된"), (7, "골격이다.")], "ko") is None

def test_shape_english_dangling_and_lone_conjunction():
    assert "dangling" in shape_check([(2, "Based on this,"), (9, "is one of the sources, so"), (12, "I introduced it.")], "en")
    assert shape_check([(1, "But"), (2, "even this"), (7, "I can't.")], "en") is None
    assert "splits a word" in shape_check([(2, "the son-in-law"), (3, "'s"), (5, "sentence was light.")], "en")

def test_glues_needs_bare_noun_before_and_particle_after():
    assert join_texts(["응답률", "도", "더 높습니다."]) == "응답률도 더 높습니다."
    assert join_texts(["팔과 턱", "에 상처를 입었고,"]) == "팔과 턱에 상처를 입었고,"
    assert join_texts(["빠르게", "이 숫자를"]) == "빠르게 이 숫자를"
    assert join_texts(["그래서,", "이 GU 종양 영역에서"]) == "그래서, 이 GU 종양 영역에서"
    assert glues("그래서,", "도") is False

def test_fix_leading_punct_and_merge_same_after():
    assert fix_leading_punct([(9, "there is a bomb"), (10, ", namely,"), (15, " the spread")]) == [(9, "there is a bomb,"), (10, " namely,"), (15, " the spread")]
    items, covers = merge_same_after([(2, "또한,"), (13, "여러분의 모든 전기에"), (13, "깊은 인상을 받았습니다.")], [{1, 2}, {3}, {4}])
    assert items == [(2, "또한,"), (13, "여러분의 모든 전기에 깊은 인상을 받았습니다.")] and covers == [{1, 2}, {3, 4}]
```

`test_build_prefix14.py`:
```python
def test_delta_glues_particle_piece():
    assert _delta([(4, "응답률")], [(6, "도")]) == "도"
    assert _delta([(4, "응답률도")], [(7, "더 높습니다.")]) == " 더 높습니다."
```

`test_label_schedule.py`:
```python
def test_rules_and_cap_flags_reach_the_prompt(tmp_path):
    rules = tmp_path / "r.txt"; rules.write_text("RULE-MARKER {N}", encoding="utf-8")
    RULES["path"] = str(rules); DEFERRAL_CAP["en"] = DEFERRAL_CAP["ko"] = 0
    assert "RULE-MARKER 1000000" in build_prompt(ROW, "") and n_cap("en") == 10**6
```

- [ ] **Step 2: 실패 확인** — `env -u PYTHONPATH .venv/bin/python -m pytest -q -p no:cacheprovider core/research/zprobe/labels -k "shape or glues or leading or delta or rules_and_cap"` → `'s` 검사 테스트 1개 FAIL, 나머지는 PASS 여야 한다(이미 구현됨).
- [ ] **Step 3: `shape_check` 에 영어 어절 쪼갬 거부 추가** — `schedule_gate.py` 의 영어 분기 앞에: 조각이 `re.match(r"^\s*'(s|re|ve|ll|d|m)\b|^\s*n't\b", t)` 면 `shape: piece {i+1} "{t}" splits a word - join it to the previous piece`. 마지막 조각도 검사한다(쪼갬은 위치와 무관).
- [ ] **Step 4: 전체 테스트 통과 확인** — `env -u PYTHONPATH .venv/bin/python -m pytest -q -p no:cacheprovider core/research/zprobe/labels` → 전부 PASS.
- [ ] **Step 5: 커밋** — `test(zprobe): prefix15 형태 게이트·붙이기 테스트, 영어 어절 쪼갬 거부`

### Task 2: 판정 탈락분 재라벨 입력 — `label_schedule.py --only-sids --notes`, `judge_schedule.py` 테스트

**Files:**
- Modify: `core/research/zprobe/labels/label_schedule.py` (인자 둘, `label_one` 의 첫 요청 메모)
- Modify: `core/research/zprobe/labels/judge_schedule.py` (`verdict` 의 malformed 처리)
- Create: `core/research/zprobe/labels/test_judge_schedule.py`
- Modify: `core/research/zprobe/labels/test_label_schedule.py`

**Interfaces:**
- Consumes: `label_one(row, hy_row, ask, fail=None, note="")` — `note` 가 있으면 1차 요청부터 프롬프트 끝에 붙는다(지금은 재요청에만 붙음).
- Produces: `label_schedule.py --only-sids FILE`(한 줄에 sid 하나; 그 sid 만 처리) `--notes FILE`(judge jsonl; `pass` 가 거짓인 행의 `note` 를 `"A reviewer found: {note}. Fix that."` 로 1차 요청에 붙임). `judge_schedule.verdict(reply) -> (defects: list[str], note: str) | None`; main 은 `None` 이면 한 번 재요청하고 또 None 이면 `{"defects": [], "note": "malformed", "pass": true, "unjudged": true}` 로 쓴다 — 탈락이 아니다(Review Focus 5).

- [ ] **Step 1: 실패하는 테스트**

`test_judge_schedule.py`:
```python
def test_verdict_codes_only():
    assert verdict({"defects": ["grammar", "style", "omission"], "note": "x"}) == (["grammar", "omission"], "x")
    assert verdict({"defects": "grammar"}) is None and verdict({}) is None

def test_prompt_mentions_reference_only_when_present():
    row = {"lang": "ko", "tgt": "en", "src": "아피아는", "ref": "Apia is"}
    assert "Reference translation" in build_prompt(row, "Apia is.") and "Reference translation" not in build_prompt({**row, "ref": ""}, "Apia is.")
```

`test_label_schedule.py`:
```python
def test_note_goes_into_first_request():
    ask = fake([GOOD])
    label_one(ROW, HY, ask, note="A reviewer found: missing subject. Fix that.")
    assert ask.calls[0].endswith("A reviewer found: missing subject. Fix that.")
```

- [ ] **Step 2: 실패 확인** — `pytest … -k "verdict or reference or note_goes"` → FAIL (`note` 인자 없음, `build_prompt` 미정의).
- [ ] **Step 3: 구현** — `label_one(…, note="")`: `note` 초기값을 인자로. `main`: `--only-sids`, `--notes` 파싱; `todo` 를 sids 로 거르고 `work(row)` 가 `label_one(…, note=notes.get(sid, ""))`. `judge_schedule.main`: malformed 1회 재요청, 그래도 None 이면 `unjudged` 행.
- [ ] **Step 4: 통과 확인** — 전체 라벨 테스트 PASS.
- [ ] **Step 5: 50문장 파일럿으로 흐름 확인($0.05)** — `pilot_v5.judge3.jsonl` 의 탈락 11 sid 를 `--only-sids`/`--notes` 로 재라벨 → 새 파일을 판정 → 통과 수를 적어 둔다(보고용).
- [ ] **Step 6: 커밋** — `feat(zprobe): 판정 탈락 문장만 메모 붙여 재라벨 (--only-sids/--notes), 판정 malformed 는 미판정`

### Task 3: 판정 합치기 `compose_judged.py`

**Files:**
- Create: `core/research/zprobe/labels/compose_judged.py`
- Create: `core/research/zprobe/labels/test_compose_judged.py`

**Interfaces:**
- Consumes: S1 라벨 jsonl, J1 판정 jsonl, S2 재라벨 jsonl, J2 판정 jsonl (파일 형식은 Task 2 의 출력), `.fail.jsonl` 둘.
- Produces: `compose(s1, j1, s2, j2) -> (kept: list[rec], rescue: list[{"sid", "why"}])` 와 CLI `--s1 --j1 --s2 --j2 --fail1 --fail2 --out schedule.judged.jsonl --out-rescue rescue.jsonl`. 규칙: J1 통과(또는 `unjudged`) → S1 행; J1 탈락이면 S2 행이 J2 통과일 때만 S2 행; 그 외(J2 탈락, S2 없음, FAIL) → rescue 에 `why` ∈ {`judge`, `fail`}. 한 sid 는 한 곳에만(Review Focus 4). 출력 행에 `judge_attempts: 1|2` 를 남긴다.

- [ ] **Step 1: 실패하는 테스트** — `test_compose_judged.py`: 위 네 경우(통과/재라벨 통과/재라벨 탈락/FAIL) 각 sid 하나씩 만들어 `compose` 결과의 sid 집합과 `why` 를 단언. 어느 쪽에도 없는 sid 가 0 임을 단언.
- [ ] **Step 2: 실패 확인** → 모듈 없음.
- [ ] **Step 3: 구현** — 순수 함수 + 파일 읽는 `main`. 같은 sid 가 S1 에 두 번 있으면 마지막 행.
- [ ] **Step 4: 통과 확인.**
- [ ] **Step 5: 커밋** — `feat(zprobe): 1차·재라벨 판정 결과를 하나의 라벨 파일로 합친다`

### Task 4: 완결성 게이트 + 기다림 라벨 + 위생표 새 열 — `schedule_stats.py`, `auto_go.py`

**Files:**
- Modify: `core/research/zprobe/labels/schedule_stats.py` (`gate` 반환 확장, `make_wait_labels`, `stats_for` 새 열, `--rescue` 인자들)
- Modify: `core/research/zprobe/labels/auto_go.py` (`--dump-enko 0.15` 기본, `--rescued-max 0.25`, 기다림 행 분리)
- Modify: `core/research/zprobe/labels/test_schedule_stats.py`, `core/research/zprobe/labels/test_auto_go.py`

**Interfaces:**
- Consumes: Task 3 의 `schedule.judged.jsonl` + `rescue.jsonl`; `sentences.jsonl`(ref·src·source), `hy_prefixes.jsonl`(`prefixes[-1]` = Hy 전체 번역), `schedule_gate.clause_ends`, `join_texts`.
- Produces:
  - `gate(rows, sents, hy, comet_min, kiwi_margin) -> (keep, dropped_rows, hy_kiwi: dict[sid, float])` — 탈락 행과 CoVoST Hy kiwi 점수를 돌려준다.
  - `make_wait_labels(sids, sents, hy, hy_kiwi, kiwi_floor) -> list[rec]`: `{"sid","lang","tgt","schedule":[[K, text]], "rescue":"wait", "why": …}`; 원문 절 경계 수(`clause_ends`)와 전체 번역의 `.?!` 수가 같고 2~3 이면 절 단위로 나눈다(after = 각 절 끝 인덱스, text = 해당 문장). 참조 없고 `hy_kiwi[sid] < kiwi_floor` 면 제외하고 `why: "hy_kiwi_floor"` 로 `--out-dropped` 에 쓴다.
  - `stats_for(rows, …)` 새 키: `piece_len_median`, `short_share`(3자 이하, 조각이 접속사·감탄사 하나뿐이면 제외 — kiwipiepy 태그 MAJ/IC 또는 영어 and/or/but/so/oh/well), `pieces_per_word`, `single_share`. 기다림 행(`rescue=="wait"`)은 `stats_for` 에 넣지 않고 별도 키 `rescued_wait: {"n", "share", "by_why": {...}}` 로 표에 찍는다. `judge_fail`·`retried` 는 입력 파일(J1 과 `judge_attempts`)에서 센다.
  - CLI: `--rescue rescue.jsonl --judge j1.jsonl --kiwi-floor-pct 10 --out-dropped dropped.jsonl`; `--out-gated` 에 통과 행 + 기다림 행을 함께 쓴다.
  - `auto_go.py`: 기다림 행은 분리해 `rescued share > --rescued-max` 면 NO; `--dump-enko` 기본 0.15.

- [ ] **Step 1: 실패하는 테스트** — `test_schedule_stats.py`: `make_wait_labels` 가 (a) 참조 있는 문장을 `[[K, ref]]` 로, (b) 절 2개·마침표 2개 문장을 2항목으로, (c) CoVoST 를 Hy 번역으로, (d) kiwi 바닥 미만을 제외하는지; `stats_for` 가 `short_share` 에서 `그리고` 단독을 빼는지, `rescue` 행을 받으면 ValueError 를 내는지(섞이지 않게). `test_auto_go.py`: 기다림 30% 면 `AUTO_GO_NO … rescued`.
- [ ] **Step 2: 실패 확인.**
- [ ] **Step 3: 구현** — 기존 `gate` 호출부와 표 출력은 유지, 새 열은 `print_table` 의 열 목록에 추가. COMET 로딩은 기존처럼 `.venv-tpm` 에서만 — CPU 테스트는 `--skip-completeness` 경로.
- [ ] **Step 4: 통과 확인** + 파일럿 50문장으로 실행 1회(`.venv-tpm`, GPU 필요 — 4090 에서; 여기선 `--skip-completeness` 로 형식만).
- [ ] **Step 5: 커밋** — `feat(zprobe): 기다림 라벨 구제와 위생표 새 열(조각 길이·접속사 제외 3자 비율·구제율·판정 탈락률)`

### Task 5: 체인 `prefix15_chain.sh` 와 설정

**Files:**
- Create: `core/research/zprobe/train/prefix15_chain.sh`
- Create (gitignore, 4090 `p13_scripts/cfg_backup/` 에도 복사): `configs/pipelines/brig.koen+prefix15-lidreset.yml` — `brig.koen+prefix14-lidreset.yml` 에서 `adapter.model` 만 `…/merged/prefix15_full`, meta description/tags 를 prefix15 로.

**Interfaces:**
- Consumes: Task 2·3·4 의 CLI. prefix14 체인의 `roomy`/`stage`/`submit`/`wait_job`/post 단계를 그대로 가져온다(`BENCH=0` 지원).
- Produces: 단계와 표시 파일(모두 `$OUT=models/zprobe/data/prefix15`):
  1. `label` → `schedule_gpt-6-luna.jsonl` (+`.fail.jsonl`), `label.done`
  2. `judge1` → `schedule.judge1.jsonl`, `judge1.done`
  3. `relabel` → `schedule.retry.jsonl` (`--only-sids` = judge1 탈락 sid, `--notes` judge1), `relabel.done`
  4. `judge2` → `schedule.retry.judge2.jsonl`, `judge2.done`
  5. `compose` → `schedule.judged.jsonl`, `rescue.jsonl`
  6. `stats` (`.venv-tpm`, `roomy 6000`) → `schedule.gated.jsonl`(통과+기다림), `dropped.jsonl`, `logs/prefix15_stats.log`, `stats.done`; `auto_go` → `GO`
  7. `rows` → `train_prefix15.jsonl`/`val_prefix15.jsonl`
  8. `sft` (`roomy 20000`) → `adapters/prefix15/final`; `merge` → `merged/prefix15_full`; `P15_SFT_READY`
  9. `bench` (매니저 queue, `BENCH=0` 이면 생략) → `post`(en rescore·COMET·lookahead·요약 출력) → `P15_DONE`
  마커 접두사 `P15`. 단계마다 "파일 있으면 건너뜀". 각 라벨·판정 단계 뒤에 누적 비용(usage 합)을 로그에 찍는다.

- [ ] **Step 1: 스크립트 작성** — 위 단계 순서대로. 라벨 단계는 `rc=3`(API 오류 >1%) 이면 10분 뒤 1회 재실행(prefix14 와 같음). `STOP_AFTER_ROWS=1` 지원.
- [ ] **Step 2: 문법·드라이런** — `bash -n`; 그리고 `OUT` 을 임시 디렉터리로 바꿔 파일럿 50문장(`pilot_sentences.jsonl`)에 `STOP_AFTER_ROWS=1 BENCH=0` 으로 1~5 단계 실행(GPU 없는 머신에선 6단계가 `roomy` 에서 멈추므로 거기까지; $0.1).
- [ ] **Step 3: 커밋** — `feat(zprobe): prefix15 체인 — 라벨·판정·재라벨·구제·위생표·SFT·벤치`

### Task 6: 4090 실행과 보고

**Files:**
- Modify: `docs/superpowers/specs/2026-11-12-prefix15-closed-unit-schedule-design.md` §6 에 본 라벨 위생표 결과 한 단락 (실험 기록이므로 추가만).

**Interfaces:**
- Consumes: origin `exp/schedule-labels` 최신, 4090 의 `/home/skkai/STiTy-wt/p13`(prefix14 데이터 전부 있음), `.venv`(conda stity)·`.venv-tpm`(+kiwipiepy), `.env` 의 `LETSUR_API_KEY`.

- [ ] **Step 1: 4090 준비** — `git pull`, `.venv/bin/python -m pip install kiwipiepy openai pytest` 필요 시, 라벨 테스트 통과 확인, `mkdir -p models/zprobe/data/prefix15`.
- [ ] **Step 2: 체인 띄우기** — `tmux new-session -d -s p15 -c /home/skkai/STiTy-wt/p13 "bash core/research/zprobe/train/prefix15_chain.sh >> /tmp/p15_chain.log 2>&1"` (매니저가 안 떠 있으면 `BENCH=0`).
- [ ] **Step 3: 위생표 확인** — `P15_STATS_READY` 뒤 `logs/prefix15_stats.log`: 방향별 조각 길이·3자 비율·dump·first_after·구제율·판정 탈락률. `AUTO_GO_NO` 면 표를 보고 판단 — 기준은 설계 §3.5, 납득되면 `touch $OUT/GO` 뒤 재실행.
- [ ] **Step 4: 비용 보고** — 라벨·판정·재라벨 usage 합계와 예산 대비.
- [ ] **Step 5: 결과 보고** — SFT eval_loss/eval_cer·행 수·소요, 벤치 ko/en COMET·BLEU·LongYAAL·lookahead·반복 수를 prefix13tr·prefix14 와 나란히. 목표(§2) 대비 판정. GRPO 는 목표를 넘겼을 때만 다음 단계로.
