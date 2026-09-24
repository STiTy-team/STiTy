# bench/ — 설정 파일로 도는 벤치마크

**파이프라인 설정과 데이터셋 설정 두 개가 한 실행을 정의한다.** 둘 다 저장소 루트
`configs/` 에 있고, 이름으로 골라 합친다. 파이프라인 설정에 적힌 이름으로 레지스트리에서
부품을 조립한다.

**WebSocket 서버를 띄우지 않는다** — VAD·커밋 로직·번역기를 전부 그대로 쓰되 한 프로세스
안에서 직접 조립한다.

```bash
# 데이터셋은 별도 리포다. 체크아웃한 디렉토리가 곧 STITY_DATA_ROOT 다.
git clone git@github.com:STiTy-team/datasets.git ~/datasets
export STITY_DATA_ROOT=~/datasets
bash $STITY_DATA_ROOT/fleurs/install.sh        # 내려받고 변환까지

make bench CONFIG=baseline DATASET=fleurs-en-ko     # 실행
make replay                                         # bench/runs/ 중 가장 최근 실행을 본다
make replay RUN=baseline-fleurs-en-ko TOPK=20       # 실행과 개수를 직접 고른다
```

**bench 는 자기 uv 환경에서 돈다.** [uv](https://docs.astral.sh/uv/) 만 설치돼 있으면 된다 —
`make` 가 `uv run --project bench` 로 부르고, uv 가 `bench/pyproject.toml`·`bench/uv.lock` 대로
`bench/.venv` 를 만든다(처음 한 번 수 GB). 저장소 루트의 `.venv` 나 다른 파이썬 환경은 건드리지
않는다. 의존성을 바꾸려면 `bench/pyproject.toml` 을 고치고 `uv lock --project bench` 로 잠근다.

`make replay` 는 http://localhost:9130 에 페이지 하나를 띄운다. 헤더의 드롭다운으로
`bench/runs/` 에 있는 다른 실행으로 서버를 다시 켜지 않고 바로 옮겨 갈 수 있다.

**항목 하나가 재생 단위다 — 실행 전체를 이어 붙이지 않는다.** 왼쪽 목록에서 항목을
고르면 그 항목만 자기 시계(0초부터)로 재생된다. 목록 맨 위는 실패·빈 전사 항목 전부와
최악 10개(기본값, `--top-k`/`TOPK=` 로 바꾼다)다. 최악은 `wer` 로 고르고, 전사 참조가 없는
항목은 `sentence_bleu` 로, 참조가 아예 없으면 `avg_fsl_sec` 로 고른다(`replay.RANKING`).
나머지 항목도 목록 아래에 전부 있다. **페이지에는 첫 항목의 이벤트만 실리고** 다른 항목은
고를 때 서버에서 받는다 — 발표 하나의 이벤트만 수 MB 라 여러 개를 한 페이지에 실으면 멎는다.

항목마다 **참조**(전사와 번역)가 "Reference" 칸에, 그 항목의 언어 쌍이 화면 위 두 알약에
나온다. 발표 단위(`longform`) 항목은 발표 오디오 전체를 재생한다.

**"On the phone" 옆 두 번째 드롭다운은 다른 실행과 나란히 본다.** 목록은 지금 보는
실행과 `manifest.jsonl` 이 같은(=항목 id 가 겹치는) 실행만 나온다 — 다른 데이터셋으로
돌린 실행은 애초에 항목 id 가 안 맞아 비교가 안 된다. 고르면 그 실행에서 **지금 보는
항목 하나만** 서버에 따로 요청한다: top-10 밖이라 그 실행 자신의 페이지에는 안 실렸을
수도 있는 항목이라서다. 두 번째 화면은 시계를 공유하고(같은 오디오니까) 소리는 첫
화면 것만 낸다 — 같은 클립을 두 번 겹쳐 듣는 건 비교에 도움이 안 된다.

고른 항목 안에서는 왼쪽이 휴대폰이 `final` 로 그린 화면 그대로(번역이 본문, 전사가 그
아래), 오른쪽은 **휴대폰에는 안 보이는** 그 항목의 이벤트 줄기 전부다. 둘 다 같은 시계
— 이벤트의 `audio` 위치 — 가 움직인다.

그 아래가 **레이어 타이밍**이다. 커밋 하나가 한 줄이고, 줄 안에서 레인 하나가 시간이
걸린 구간 하나다 — 오디오가 들어온 구간, 그 위에 무엇이 얼마나 돌았는지, 커밋이 정해진
순간(◆)과 그때의 FSL 이 한 눈에 겹쳐 보인다. 가로축은 **벽시계**(그 항목의 첫 청크부터의
초)라 "오디오보다 얼마나 밀렸나"가 거리로 읽힌다. 레인 목록은 그 실행에 실제로 나온
타입에서 만들어진다 — §시간은 자동으로 재진다.

데이터셋은 `fleurs` 와 `acl6060` 둘이다. 계약(`dataset.yml` + `manifest.jsonl`)과 새
코퍼스 붙이는 법은 그 리포의 README 에 있다. bench 에는 데이터셋별 분기가 없다.

## 설정

**설정 파일은 두 종류뿐이고 저장소 루트 `configs/` 에 있다.** 파이프라인 설정이 *무엇을
재는가*(부품과 커밋 정책)를, 데이터셋 설정이 *무엇을 먹이는가*(코퍼스와 방향)를 적는다.
실행마다 파일을 새로 만들지 않는다 — 있는 둘을 이름으로 골라 합친다.

```bash
python -m bench --config baseline --dataset acl6060-en-de
```

`configs/pipelines/baseline.yml` — **재는 대상 그 자체다.** 무엇이 오디오를 넣어 주는지는
여기 없고, 그래서 이 파일은 bench 전용이 아니다.

```yaml
pipeline:
  name: cascade
  transcription:
    name: qwen3
    chunk_size_sec: 2.0      # 모델별 설정은 그 모델 밑에 둔다
    max_new_tokens: 128
  translation:
    name: local
    model: google/madlad400-3b-mt
  vad:
    name: silero
    min_silence_ms: 800
commit: seg                  # seg | punct | always
gpu_memory_utilization: 0.5
```

`configs/datasets/acl6060-en-de.yml` — 코퍼스와 방향이 한 쌍이다. 한 코퍼스를 다른 방향으로
재려면 파일을 하나 더 둔다 (`fleurs-en-ko`, `fleurs-en-de`, …). 파일 이름이 곧 그 쌍의 이름이다.

```yaml
dataset:
  name: acl6060
languages:
  lang: en
  target: de
```

데이터셋이 크면 `limit: N` 으로 앞의 N개만 돈다(`longform` 이면 발표 N개).

```yaml
dataset:
  name: covost2
  limit: 200
```

### 발표를 통째로 흘리기 (`longform`)

```yaml
dataset:
  name: acl6060
  longform: true
```

기본은 항목(문장)을 하나씩 흘린다. `longform: true` 면 같은 `group` 의 항목들을 발표 하나로
묶어 **그 오디오 파일을 처음부터 끝까지 한 세션으로** 흘린다. 항목들은 버려지지 않고 그
발표의 참조 분절이 된다 — 행의 `reference_segmentation`(문장마다 `offset`·`duration`·전사·번역)이
IWSLT 의 segmentation yaml 과 같은 것이다. 그래서 그룹의 항목이 모두 한 오디오 파일 안의
`offset` 구간이어야 한다(`acl6060`·`tedlium`). 아니면 시작 전에 죽는다.

행 하나가 발표 하나라 문장 단위 지표는 달라진다.

- `longyaal_ms`·`longyaal_ca_ms` 가 나온다. OmniSTEval 이 가설을 참조 문장에 재분절한 뒤
  `is_longform=True` 로 YAAL 을 낸다 — IWSLT 2026 이 지연 기준으로 쓰는 값이다.
- `bleu`·`comet` 은 같은 재분절 결과로 문장마다 한 쌍씩 낸다. COMET 의 원문은 그 문장의 참조 전사다.
- `laal`·`yaal` 은 빠진다. 발표 전체를 문장 하나로 보는 값이라 의미가 없다.
- `wer`·`cer`·`fsl`·`token_emission` 은 발표 전체에서 그대로 나온다. 정렬 시각은 문장
  `offset` 만큼 밀어 발표 시각으로 바꾼다.

`configs/datasets/acl6060-en-de-longform.yml` 이 이 설정이다.

**데이터셋은 통째로 돈다.** 개수를 줄이는 설정은 없다 — 일부만 돈 결과와 전부 돈 결과가
같은 이름으로 같은 디렉토리에 쌓이면 나중에 어느 쪽인지 알 수 없고, 그 둘은 비교도 안 된다.
빨리 보려면 항목이 적은 데이터셋을 따로 둔다.

**이름이 틀리면 있는 이름들을 알려주고 죽는다.** 설정 안의 오류도 그 설정 파일 기준 경로로
나온다 — `commit: unknown mode 'nope'`, `dataset: unknown key(s) ['nmae']`.

**부품은 파이프라인 안에 쓴다.** transcription·translation·vad 는 파이프라인의 인자이지
형제가 아니다. 다른 파이프라인이 쓰지 않는 부품을 받는 파이프라인도 자기 options 에 적으면
되고, 바깥이 그 종류를 먼저 알아야 할 일이 없다.

`translation: gpt` 처럼 문자열만 쓰면 `{name: gpt}` 로 풀린다.

**오디오를 어떻게 먹이는지는 설정이 아니다.** 청크 200ms, 클립 뒤에 붙이는 침묵 4000ms,
실시간 속도 — 셋 다 `__main__.py` 의 상수다. 이건 클라이언트를 기술하는 값이지 재는
대상이 아니고, 실행마다 다르게 먹인 두 결과는 애초에 비교가 안 된다. 쓰인 값은
`summary.json` 의 `pacing` 에 남는다.

**모델별 설정은 그 모델 밑에 쓴다.** 청크 크기나 토큰 예산은 체크포인트에 딸린 값이라
최상위로 평평하게 펴지 않는다. 컴포넌트가 모르는 키를 주면 **에러로 죽는다** — 조용히
버려지면 `max_new_tokens: 256` 을 적어 놓고 아무 일도 안 일어난 채 그럴듯한 숫자가 나온다.

### `commit` 은 명시적으로 풀린다

| `commit` | `always_commit` | `enable_dot_commit` | `hide_seg` |
|---|---|---|---|
| `seg` | false | false | false |
| `punct` | false | true | true |
| `always` | true | false | true |

프로덕션 서버는 `enable_dot_commit` 기본값을 **가중치 경로에서 유도한다**
(`_infer_dot_commit_default`). 그래서 같은 명령이 체크포인트에 따라 다른 정책으로 돈다.
bench 는 `parse_args` 를 부르지 않고 위 표로만 푼다. 해석된 값은 결과 파일에 전부 남는다.

`hide_seg` 는 SEG 를 뱉는 가중치로 `punct`/`always` 축을 돌릴 때 필요하다. 없으면 축이
조용히 섞인다 — 실측으로 punct 축 커밋의 36%가 seg 였고, 그 커밋만 확정 게이트를 건너뛰어
지연이 실제보다 좋게 잡혔다.

### 한 실행은 한 방향이다

`lang` ↔ `target` 한 쌍이다. 서버의 방이 두 언어를 갖는 것과 같다 — `lang` 으로 들린 말은
`target` 으로, `target` 으로 들린 말은 `lang` 으로 번역한다. 어느 쪽으로 들렸는지는 파이프라인이
정한다: bench 는 이 쌍만 넘기고(`LanguagesConfig.target_for`), **항목의 실제 언어는 넘기지 않는다.**
넘기면 언어 판정을 정답으로 대신하게 되어 `routing` 지표가 늘 1.0 근처로 나온다.

`lang`/`target` 은 그대로 ASR 의 `allowed_languages` 가 되어 언어 이름 토큰에 로짓 바이어스를
건다. 즉 **언어 설정은 번역만이 아니라 WER 도 바꾼다.**

거는지 마는지는 **파이프라인 쪽 설정**이다 — `transcription: {restrict_languages: false}`
로 끈다. 디코딩을 제약하는 값이라 재는 대상에 속하고, 어느 언어로 제약할지는 실행이
시작될 때 정해진다. bench 는 데이터셋 설정의 `lang`/`target` 을 넘기고, 서버는 클라이언트가
접속하며 고른 두 언어를 넘긴다.

`routing` 지표가 언어 판정과 라우팅을 잰다. BLEU·COMET 은 **올바르게 라우팅된 세그먼트만**
으로 낸다. 한국어 출력을 프랑스어 참조와 비교하면 번역 품질 문제와 언어 판정 문제가 한
숫자에 섞여 둘 다 못 읽는다.

## 나오는 것

**설정 한 쌍이 디렉토리 하나다.** 이름은 `<파이프라인>-<데이터셋>` 이고 타임스탬프가 붙지
않는다. 같은 쌍을 다시 돌리면 그 디렉토리를 **덮어쓴다** — 그 쌍의 현재 답이 하나만
남는다는 뜻이다. 지우려면 그 디렉토리만 지우면 된다.

```
runs/baseline-fleurs-en-ko/
  summary.json                      이 실행의 요약
  items.jsonl                       항목 단위 행. append (죽어도 채점된다)
  events.jsonl                      이벤트 전부. 이게 원본이고 리플레이가 읽는 것이다
```

실행이 시작할 때 위 세 파일을 먼저 지운다. 스트림은 append 로 열리므로(중간에 죽어도
거기까지 채점된다) 안 지우면 지난 실행 뒤에 이어 붙어 두 실행이 한 기록으로 섞인다.

**설정은 여기 복사되지 않는다.** 입력은 `configs/` 에 있고 여기는 산출물만 둔다. 무엇을
돌렸는지는 `summary.json` 의 `config` 에 두 파일 내용이 그대로 들어가므로 따로 복사할
이유가 없다.

| 파일 | git |
|---|---|
| `runs/<name>/summary.json` | 추적한다 (작다, 실행 비교가 diff 로 보인다) |
| 나머지 | 안 한다 |

요약과 리플레이는 이벤트 스트림을 다시 읽어 만든 **투영**이다. 전부 append 라 중간에
죽어도 그때까지가 남고 채점된다.

`summary.json` 의 `config.resolved` 에는 설정에 안 쓴 값까지 **실제로 쓰인 값**이
들어간다. 이게 없으면 두 실행을 비교할 수 없다.

## 이벤트

표준 `logging` 위에 올려서 한 줄 JSON 으로 흘린다. 시각 `t` 는 **그 항목의 첫 오디오
청크로부터의 초**다(벽시계가 아니다) — 서버가 내는 모든 시간이 그 원점에서 재므로
`fsl`/`laal` 검산이 그 시계에서만 성립한다.

서버 로그가 같은 줄기에 섞인다. 그래서 서버가 `send_message` 전에 버리는 커밋
(`[SILENCE-DROP]`, `[HALLUC-DROP]`, `[EMPTY-DROP]`, `[TAIL-DROP]`, `[AST-LATE]`)이
**공짜로 보인다** — 소켓 모양의 싱크로는 아예 안 보이는 것들이다. final 이 0건인 항목의
이유를 여기서 찾는다.

빈 전사와 실패한 항목은 버리지 않는다. `summary.json` 의 `counts.empty_hypothesis` 로 세고 리플레이에 **항상**
넣는다(top-k 와 무관하게).

**`partial` 은 아직 확정되지 않은 줄이다.** 커밋만 기록하면 한 발화가 끝에서 통째로
튀어나오는 실행만 남는다 — 화면에 실제로 보이는 것도, 이 시스템이 하려는 것도 그게
아니다. 서버와 같은 모양으로 낸다: 통째로 교체할 전체 문자열, 120ms 간격 제한,
프레임마다 강제 재동기화(그게 없으면 청크마다 첫 콜백 하나 — 글자 몇 개짜리 — 만
남는다). 빈 문자열은 "지우라"는 신호이고 커밋 직후에만 나간다. 리플레이의 휴대폰
화면에서 흐린 말풍선이 이것이다.

### 시간은 자동으로 재진다

**부품에 아무것도 안 쓴다.** 레지스트리가 등록된 부품의 **프로토콜 메서드를 감싸서**
시간을 잰다 — `Transcriber` 의 `flush`·`finish`, `Translator` 의 `translate`,
`Corrector` 의 `correct`. 새 백엔드를 붙이면 레인이 그냥 생기고,
새 종류를 만들어도 그 베이스가 선언한 메서드가 곧 레인이 된다.

```jsonc
{"t": 2.0015, "type": "timing", "tag": "flush", "audio": 2.0, "dur": 0.3531}
```

**시간을 잰 줄은 `type` 이 `timing` 이고 `tag` 가 무엇을 쟀는지 말한다.** `t` 는 다른
모든 줄과 같은 뜻 — **시작한 시각**이다. 줄이 나가는 건 끝날 때지만(그래야 길이를 안다)
시각은 시작을 가리키므로 필터 칩도 콘솔의 타입 칸도 그냥 맞는다.

읽는 쪽에는 태그 목록이 없다. 규칙 한 줄이다 — **`dur` 을 달고 있으면 시간이 걸린
구간이고, 그 `tag` 가 레인 이름이다.** 레인 순서도 색도 그 실행에 실제로 나온 태그에서
만들어진다.

**레인 이름이 곧 프로토콜이라 코드와 어긋날 수가 없다.** 메서드 이름을 바꾸면 레인
이름이 따라 바뀐다. 관례를 지키라고 부탁할 자리가 없고, 지키는 걸 잊을 자리도 없다.
**막대를 열어 놓고 닫지 않을 방법도 없다** — 재는 단위가 함수라 예외로 빠져나가도
닫힌다.

안 재는 것 셋. **`load`·`close`·`start` 는 항목 시계 밖**이라 재도 길이가 안 나온다.
**`transcribe`·`detect` 는 청크마다 불린다**(`registry.PER_CHUNK`) — 200ms 마다 막대가 두 개씩
생겨 이벤트 줄기와 리플레이 표를 덮는다. 그 안에서 시간이 드는 것은 모델 호출이고, 그건
아래 `decode` 레인이 잰다.
**파이프라인은 안 잰다** — 부품을 엮는 쪽이고 그 막대는 부품 막대를 전부 덮을 뿐이다.

메서드 하나보다 잘게 재야 할 때만 직접 붙인다. 지금 트리에 한 군데 있다
(`qwen3.py` 의 `_decode`·`_decode_stream` — 전사 백엔드가 모델을 부르는 자리. `qwen-seg` 도 물려받는다).

```python
@timing.measure("decode")
async def _decode(self, state) -> None:
    await self.model.finish_streaming_transcribe(state)
```

**태그는 모델이 아니라 하는 일을 가리킨다.** `qwen3_generate` 가 아니라 `decode` 여야
다른 전사 백엔드의 같은 구간이 같은 레인에서 비교된다.

**한 레인은 그 이름의 일을 전부 덮어야 한다.** 모델 호출 중 하나만 감싸 두면 `decode`
레인은 디코딩의 일부만 보여주면서 전부인 척한다. 실제로 그랬던 적이 있어서 지금은 두
백엔드의 모델 호출이 모두 `_decode`·`_decode_stream` 을 지난다.

**채점에는 들어가지 않는다.** `summary.json` 은 `items.jsonl` 에서 나오고 이벤트 줄기를
보지 않으므로, 레인이 늘어도 지표는 그대로다.

## 로그는 넷뿐이다

`log.debug` · `log.info` · `log.warning` · `log.error`. 줄은 `[TAG] 문장` 으로 쓰고,
태그는 스트림에 나갈 때 `tag` 필드로 떨어져 나간다 — 산문을 정규식으로 되파낼 일이 없다.

```python
log.info("[COMMIT-SKIP] reason=%s text=%r", reason, shown)
```

**숫자를 문장에 녹이지 않는다.** 그림에 찍히거나 채점되는 값은 로그가 아니라 부품이
돌려주는 기록(`Transcribed`·`Partial`·`Final`)으로 다닌다. 그래서 `core/components` 의
어떤 파일도 스트림에 직접 쓰지 않는다 — `events.jsonl` 은 `__main__.py` 가 돌려받은
객체를 그대로 적는다.

## 지표

| 지표 | 비고 |
|---|---|
| `wer` | 주 숫자. `jiwer` 가 센다. 빈 가설을 전체 삭제로 센다. 채점 전에 참조와 가설 모두 Whisper 정규화(`whisper-normalizer`)를 거친다 — 영어는 숫자·축약형·철자까지 맞추는 영어 정규화(`twenty-five`→`25`, `don't`→`do not`), 나머지 언어는 소문자화와 문장부호·기호·괄호 속 이벤트 제거만 한다. 태국어·힌디어 모음 부호는 남긴다 |
| `wer_by_lang` | 원문 언어(데이터셋의 `src_lang`)별 `wer`. `wer` 은 이 값들의 단순 평균이다 — 언어마다 "단어" 크기가 달라서 단어 수를 합쳐 세면 항목이 많은 언어가 숫자를 좌우한다. 언어가 하나면 `wer` 과 같다. 모델이 판정한 언어가 아니라 참조 언어로 묶는다 — 언어를 잘못 판정한 항목도 제 언어의 오류로 남는다 |
| `wer_scored_only` | 옛 숫자(빈 가설 제외). 대조용. `wer` 과 같은 방식으로 언어별 평균이다 |
| `cer` | 문자 오류 합 / 참조 문자 합. `jiwer` 가 센다. `wer` 과 같은 정규화 뒤 공백을 지우고 센다 — 한국어 띄어쓰기는 참조마다 달라서 오류로 치지 않는다 |
| `cer_by_lang` | 원문 언어별 `cer`. `cer` 은 이 값들의 단순 평균이다 |
| `fsl` | 커밋이 오디오보다 얼마나 늦게 도착했나 = `recv_elapsed_sec − decision_audio_sec`. **기록하지 않고 유도한다** — 두 시계가 이미 있으니 파이프라인이 따로 내면 어긋날 수 있다 |
| `laal` | OmniSTEval 의 LAAL(SimulEval 구현). `decision_audio_sec` 이 `d_i` 이고, 소스 길이에서 자른다 — 끝에 붙인 무음 동안 커밋해도 발화보다 긴 지연이 나오지 않게. 벽시계 기준은 `laal_ca_ms` |
| `bleu` | 라우팅이 맞은 것만. **발화 하나가 한 쌍**이다 — 세그먼트를 다시 이어 붙여 채점한다. 아무것도 커밋하지 않은 발화는 빈 번역으로 채점한다. 언어 쌍마다 따로 채점하고(`bleu_by_pair`, 키는 `en-ko` 꼴) `bleu` 는 그 단순 평균이다 — 항목이 많은 쌍이 약한 쌍을 가리지 않게 |
| `comet` | `Unbabel/wmt22-comet-da`. 원문은 참조 전사, 번역·참조는 **BLEU 와 똑같은 문장 쌍**이다(`comet_inputs.jsonl`). 번역이 빈 문장은 모델에 넣지 않고 0점으로 센다 — 모델은 빈 번역에도 0.5 안팎을 준다. 언어 쌍마다 평균을 내고(`comet_by_pair`) `comet` 은 그 단순 평균이다. **따로 도는 단계가 채점한다** — 아래 참고 |
| `yaal_ms` | OmniSTEval 의 YAAL. LAAL 과 분모가 같고, 소스가 끝나기 **전에** 나온 단위만 센다. 첫 단위가 소스 끝 이후에 나온 항목은 채점하지 않는다. `yaal_ca_ms` 는 벽시계 기준 |
| `longyaal_ms` | `longform` 실행에서만. OmniSTEval 이 가설을 참조 문장에 재분절(SoftSegmenter)하고 문장마다 YAAL 을 `is_longform=True` 로 낸 평균이다 — 문장 끝을 넘겨 나온 단위도 녹음이 끝날 때까지는 센다. 녹음 끝은 발표 오디오 길이다 — 마지막 참조 문장의 끝으로 잡으면 발화가 끝난 뒤 커밋되는 마지막 문장이 통째로 빠진다. `decision_audio_sec` 기준이고 벽시계 기준은 `longyaal_ca_ms`. 위 '발표를 통째로 흘리기' 참고 |
| `token_emission_ms` | 단어가 실제로 발화된 끝 시각부터 화면에 **바뀌지 않고 남은** 채로 처음 나타난 시각까지. 참조와 맞게 인식된 단어만 센다. `audio` 시계 기준이고, 벽시계 기준은 `token_emission_ca_ms`. 둘 다 평균·`_p50_ms`·`_p90_ms`. 발화 시각은 데이터셋의 `alignment.jsonl` 에서 온다 — 아래 참고 |
| `commit_reasons` | 사유별 비율. `finish` 비율이 크면 축의 커밋 경로가 안 도는 것이다 |
| `lang_detect_accuracy`·`route_accuracy`·`confusion` | 언어 판정 정확도, 라우팅 정확도, 혼동 행렬. 항목별로는 `route_errors` |

**COMET 은 별도 환경에서 돈다.** `unbabel-comet` 은 `protobuf<5`·`numpy<2` 를 고정하고 vLLM 0.14 는
`protobuf>=6.30` 을 요구해서, 한 환경에 둘을 같이 풀 수 있는 버전이 없다. 그래서 `bench/comet/` 이 자기 uv 환경을 갖고, `make bench`
가 실행이 끝난 뒤 이어서 부른다. 무엇을 채점할지는 bench 가 정한다 — `metrics.translation_sentences`
가 BLEU 에 쓰는 문장 쌍을 원문과 함께 `comet_inputs.jsonl` 로 떨구고, `bench/comet` 은 그 파일만
읽어 한 번에 채점해 그 실행의 `summary.json` 에 `comet`·`comet_by_pair` 를 더한다 — 항목 수만큼
도는 게 아니라 실행당 한 번이다. 그래서 라우팅·재분절·괄호 속 이벤트 제거 규칙이 두 환경에 따로 있지 않다.
`python -m bench` 만 부르면 `comet` 은 `unavailable` 에 그 이유로 남는다. 지난 실행을 다시
채점할 때는 그 한 줄만 부른다.

```bash
uv run --project bench/comet python -m bench.comet bench/runs/baseline-fleurs-en-ko
```

처음 한 번은 uv 가 환경(PyTorch 포함 수 GB)과 COMET 모델(약 2.3 GB)을 받는다.

**단어가 언제 발화됐는지는 데이터셋의 `alignment.jsonl` 에서 온다.** 데이터셋 리포의
`convert.py` 가 변환 끝에 강제정렬기로 만든다(그 리포 README 의 "정렬"). bench 는 파일이 있으면
읽고, 없으면 토큰 방출 지연만 빠진다. 전사가 바뀐 뒤 다시 정렬하지 않은 항목은 쓰지 않는다. 정렬은 재는
대상과 무관하다 — 참조 오디오와 참조 전사만 보므로 어떤 ASR 모델을 재든 같은 시각을 쓴다.
클립 길이를 정렬 간격(80ms) 넘게 벗어난 시각은 쓰지 않는다 — ACL6060 에서 19개 단어가 그랬다.

화면에 보인 시각은 `items.jsonl` 의 `emissions` 에 있다 — 전사기가 낸 `Partial` 과 `Transcribed`
를 순서대로, 그때의 `t`·`audio` 와 함께. 화면 = 확정된 전사 + 지금의 partial 이고, 단어의
시각은 그 단어(와 그 앞 전부)가 그 뒤로 끝까지 안 바뀐 첫 순간이다. 잠깐 틀렸다가 고쳐진 단어는
고쳐진 순간부터 센다. 번역을 기다리지 않는다 — ASR 지표다.

`wer` 과 `wer_scored_only` 를 둘 다 내는 이유: 기존 `compute_wer_for_rows` 는 가설이 빈
행을 버리고(`scoring.py:19`), `process_batch` 가 그 행을 또 버린다. 실패한 발화가 두 번
숨어서 점수가 좋아 보인다. 실측으로 한쪽이 완전 실패인 두 발화에서 0.0 과 0.5 가 갈린다.

**지표는 고르지 않는다. 매 실행이 낼 수 있는 것을 전부 계산한다.** 설정으로 부분집합을
고르는 길은 없다 — 이미 GPU 시간을 치르고 얻은 숫자를 가릴 뿐이다.

**데이터가 못 받치는 지표는 없는 채로 둔다. 실패하지 않는다.** 값이 아예 빠지고
`diagnostics.unavailable` 에 이유가 남는다. `null` 은 나오지 않는다 — 지표는 숫자이거나
이유이고, 읽는 사람이 빈칸을 추측할 일은 없다.

| 데이터 | 나오는 것 | 빠지는 것 |
|---|---|---|
| 오디오만 (전사·번역 참조 없음) | `fsl`·`commit_reasons`·`routing` | `wer`·`cer`·`token_emission`·`bleu`·`comet`·`laal`·`yaal` |
| 전사는 있고 번역 참조 없음 | 위 + `wer`·`cer`·`token_emission` | `bleu`·`comet`·`laal`·`yaal` |

참조 번역이 없으면 `laal`·`yaal` 도 빠진다. 분모가 `max(|Y_hyp|, |Y_ref|)` 라서 `|Y_ref|` 를 빼면
근사가 아니라 **다른 지표(AL)** 가 되고, AL 은 짧게 생성할수록 점수가 좋아지는 구멍이 있다.
그걸 `laal_ms` 칸에 적으면 비교가 불가능한 두 숫자가 한 열에 섞인다.

## 구조

```
__main__.py   CLI + 실행. ASR 서버를 import 하는 유일한 모듈이고, 그 import 는
              Engine 안에서 늦게 일어난다 — 설정 오류는 모델을 올리기 전에 걸린다
config.py     데이터셋 설정을 읽고 파이프라인 설정과 합친다. 점수를 바꾸는 값 전부 명시
              해석. 파이프라인 설정 자체는 `core/config.py` 가 읽는다 — 서버와 공유한다
report.py     이벤트 스트림 → summary.json
replay.py     이벤트 스트림 → :9130 웹 페이지 (replay.html 이 화면 전부)
dataset.py    dataset.yml + manifest.jsonl (+ 있으면 alignment.jsonl) 읽기
metrics.py    items.jsonl 행 → 지표. COMET 은 입력(comet_inputs.jsonl)만 여기서 만들고 채점은 bench/comet/__main__.py 가 자기 환경에서 한다
```

채점은 `metrics.py` 가 한다. 지표 하나가 함수 하나이고(`wer`·`cer`·`bleu`·`fsl`·`laal`·`yaal`·
`token_emission`·`routing`·`commit_reasons`), 모두 `items.jsonl` 의 행 목록을 받는다.
항목 하나의 값은 행 하나짜리 목록으로 부른다. `score_row` 가 항목별 값을, `score_run` 이 실행
전체 값을 모은다 — 설정에서 무엇을 계산할지 고르는 자리가 없으니 고를 코드도 없다.

`config.py`·`dataset.py`·`report.py`·`replay.py` 는 `__main__.py` 를
import 하지 않는다.
그래야 GPU 없이 돌릴 수 있다.

데이터셋 리포는 반대로 **STiTy 를 import 하지 않는다.** 두 리포가 나란히 체크아웃돼
있다는 가정은 곧 깨진다.

## 아직 안 되는 것

- **동시 실행.** v1 은 순차 고정이다. `asr_lock` 이 생성을 직렬화하고 flush 가 겹친다.
- **서버가 버리는 커밋을 싱크로 잡기.** 로그로는 보이므로 이벤트 줄기에서 읽는다.
- **WebSocket 경로와의 대조 검증(S6).** 데이터와 가중치가 있는 머신에서 해야 한다.
  같은 항목을 두 경로로 돌려 **전사 문자열과 커밋 사유 분포가 같은지** 보는 단계이고,
  이걸 통과하기 전에는 bench 숫자를 보고서에 쓰지 않는다.
