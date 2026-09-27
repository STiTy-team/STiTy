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
SRC="en_us ko_kr" TGT=ko_kr bash $STITY_DATA_ROOT/fleurs/install.sh   # fleurs/en_us, fleurs/ko_kr

make bench CONFIG=asr.qwen-seg-en+mt.qwen3.5-4b DATASET=fleurs-en-ko  # en→ko
make bench CONFIG=asr.qwen-seg-ko+mt.qwen3.5-4b DATASET=fleurs-ko-en  # ko→en
make replay                                         # 대시보드: 데이터셋별로 실행들을 비교한다
make replay RUN=asr.qwen-seg-en+mt.qwen3.5-4b-fleurs-en-ko TOPK=20    # 그 실행의 세션 재생으로 바로 연다
```

**bench 는 자기 uv 환경에서 돈다.** [uv](https://docs.astral.sh/uv/) 만 설치돼 있으면 된다 —
`make` 가 `uv run --project bench` 로 부르고, uv 가 `bench/pyproject.toml`·`bench/uv.lock` 대로
`bench/.venv` 를 만든다(처음 한 번 수 GB). 저장소 루트의 `.venv` 나 다른 파이썬 환경은 건드리지
않는다. 의존성을 바꾸려면 `bench/pyproject.toml` 을 고치고 `uv lock --project bench` 로 잠근다.

`make replay` 는 http://localhost:9130 에 서버 하나를 띄운다(`bench/replay/`). 페이지는 둘이다.
포트는 `python -m bench.replay --port N` 으로 바꾼다.

**`/` 는 대시보드다 — 데이터셋 하나를 골라 그 데이터셋으로 돈 실행들을 비교한다.** 코퍼스(`fleurs`)를
고르고 그 안의 데이터셋 설정(`fleurs-ko-en` 등)을 고른다. 같은 항목·같은 목표 언어로 채점된
실행끼리만 한 화면에 모인다 — 다른 데이터셋의 숫자는 나란히 놓아도 비교가 안 된다. 데이터셋마다
실행 수가 달라도 된다. 실행 이름 `<파이프라인>-<데이터셋 설정>` 을 `configs/datasets/` 에 있는
이름으로 잘라 파이프라인 이름을 얻는다.

- **Leaderboard** — `summary.json` 의 지표를 종류별(전사 정확도·전사 지연·번역 정확도·번역 지연·
  언어 판정·분절)로 묶은 표. 열을 누르면 정렬되고, 열마다 가장 좋은 값이 초록이다. 좋은 방향
  (↓ 낮을수록, ↑ 높을수록)은 `static/common.js` 의 `SCORES` 가 정한다. 이름을 누르면 그 실행의
  세션 재생으로 간다. 아직 도는 실행(`summary.json` 이 없는)은 `running` 으로, 지금까지 끝낸 항목
  수와 함께 나온다.
- **Per-item distribution** — 평균 하나 대신 항목마다의 점수 분포를 실행마다 세로 기둥 하나로
  그린다. 수염 끝이 최솟값·최댓값, 상자가 25–75%, 굵은 선이 중앙값, 마름모가 평균이고, 옅은 점이
  항목 하나하나다(끌 수 있다). 중앙값이 좋은 실행이 왼쪽에 온다. 실행이 많으면 옆으로 넘긴다. 상자 그림만으로는 분포가 둘로
  갈렸는지 한쪽으로 쏠렸는지 안 보여서 점을 함께 그린다(raincloud plot 의 권고). BLEU 는 표가
  말뭉치 BLEU, 그림이 항목별 문장 BLEU 이고, COMET 은 문장마다의 점수(`comet_scores.jsonl`)다.
- **Accuracy vs latency** — 정확도 하나와 지연 하나를 골라 실행을 점으로 찍는다. 두 축 모두
  좋은 쪽이 오른쪽·위가 되도록 낮을수록 좋은 축은 뒤집는다. 점선은 두 축 모두에서 이기는
  다른 실행이 없는 실행들(frontier)을 잇는다. `summary.json` 의 값을 찍으므로 아직 도는 실행은
  빠지고, 빠진 실행과 이유가 그림 아래에 적힌다.

헤더의 **Models** 메뉴로 그 데이터셋의 실행을 골라 뺄 수 있다(기본은 전부). 표와 두 그림이 모두
따른다. 고른 데이터셋·모델·지표·정렬은 주소의 `#` 뒤에 남아 새로 고쳐도 유지되고 링크로 건넬 수 있다.

**`/replay?run=<실행>` 은 세션 재생이다.** 헤더의 드롭다운으로 `bench/runs/` 의 다른 실행으로
서버를 다시 켜지 않고 옮겨 가고, `← Dashboard` 로 돌아간다. 아직 도는 실행도 열린다 — 그때까지
끝난 항목이 보이고, `summary.json` 이 없으니 설정은 `configs/` 에서 다시 읽어(그렇다고 적힌다)
보이며, 데이터셋은 `items.jsonl` 의 오디오 경로에서 거슬러 찾는다. 실행 요약만 끝나야 생긴다.

**항목 하나가 재생 단위다 — 실행 전체를 이어 붙이지 않는다.** 왼쪽 목록에서 항목을
고르면 그 항목만 자기 시계(0초부터)로 재생된다. 목록 맨 위는 실패·빈 전사 항목 전부와
최악 10개(기본값, `--top-k`/`TOPK=` 로 바꾼다)다. 최악은 `wer` 로 고르고, 전사 참조가 없는
항목은 `sentence_bleu` 로, 참조가 아예 없으면 `avg_fsl_sec` 로 고른다(`replay/session.py` 의 `RANKING`).
나머지 항목도 목록 아래에 전부 있다. **페이지에는 첫 항목의 이벤트만 실리고** 다른 항목은
고를 때 서버에서 받는다 — 발표 하나의 이벤트만 수 MB 라 여러 개를 한 페이지에 실으면 멎는다.

항목마다 **참조**(전사와 번역)가 "Reference" 칸에, 그 항목의 언어 쌍이 화면 위 두 알약에
나온다. 발표 단위(`longform`) 항목은 발표 오디오 전체를 재생한다.

**헤더 카드의 "Data" 표는 그 항목에 들어간 데이터 자체다** — 실행 결과와 나란히 보라고
실행 이름·타임라인과 같은 카드에 있다. "Data" 줄을 누르면 접히고, 접었는지는 브라우저가 기억한다. 한 줄이 manifest 행 하나(섞인 대화면 turn 하나)로,
시간 구간·언어·화자·오디오 크기(peak / rms, dBFS)·참조 전사와 목표 언어 번역이 나온다.
줄을 누르면 그 turn 으로 가고, 재생 중에는 지금 turn 과 정렬된 단어가 칠해진다. turn 은
타임라인 아래에도 언어 색 띠로 깔린다. 소리가 작은 turn 에는 `quiet` 가 붙는다 — peak 가
-40 dBFS 아래이거나 그 항목에서 가장 큰 turn 보다 20 dB 이상 작을 때다. VAD 는 이런 turn 을
놓치고 ASR 은 듣는 일이 있어, partial 은 나왔는데 커밋이 사라진 경우를 여기서 먼저 본다.
접힌 칸에는 오디오 파일 헤더, `dataset.yml` 원문, manifest 원래 행과 줄 번호가 있다.
**`augment` 를 쓴 실행은 입힌 소리 그대로 들린다.** 재생 소리와 "Data" 표의 크기(peak / rms)는
원본 파일이 아니라 파이프라인이 들은 소리다. 입히기는 항목마다 늘 같으므로 replay 가 실행 설정의
`augment` 로 다시 만들고, 행에 남은 선택(파일·시작 위치·level)과 같은지 확인한다. 그 사이
`$STITY_DATA_ROOT/augment/` 의 파일이 바뀌어 다르게 나오면 다른 소리를 들려주는 대신 재생을 거부하고
이유를 "Data" 줄에 적는다.
manifest 를 못 찾는 실행(아직 도는 중이거나 summary 전에 죽은 실행)은 `items.jsonl` 의
`reference_segmentation` 으로 turn 을 채우고, 화자와 원래 행은 비운다.

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
순간(◆)과 그때의 FSL 이 한 눈에 겹쳐 보인다. ◆ 는 `Transcribed.committed_elapsed_sec`, 곧
디코딩 도중 `<SEG>` 가 나온 순간이고, 세로선은 그때까지 들어온 오디오(`decision_audio_sec`)
다. 오디오는 그 오디오를 처음 들은 커밋의 줄에, 번역처럼 시간이 걸린 일은 그 일이 끝난
뒤 처음 전달된 커밋(`Translated.translated_elapsed_sec`)의 줄에 들어간다 — 디코딩 한 번에서
커밋 둘이 나와도 번역 막대가 각자 자기 줄에 그려진다. 오디오 막대의 명암은 모델이 청크를
한 번 읽을 때마다 바뀐다. 가로축은 **벽시계**(그 항목의 첫 청크부터의
초)라 "오디오보다 얼마나 밀렸나"가 거리로 읽힌다. 레인 목록은 그 실행에 실제로 나온
타입에서 만들어진다 — §시간은 자동으로 재진다.

데이터셋은 `fleurs` 다. 계약(`dataset.yml` + `manifest.jsonl`)과 새
코퍼스 붙이는 법은 그 리포의 README 에 있다. bench 에는 데이터셋별 분기가 없다.

## 설정

**설정 파일은 두 종류뿐이고 저장소 루트 `configs/` 에 있다.** 파이프라인 설정이 *무엇을
재는가*(부품과 커밋 정책)를, 데이터셋 설정이 *무엇을 먹이는가*(코퍼스와 방향)를 적는다.
실행마다 파일을 새로 만들지 않는다 — 있는 둘을 이름으로 골라 합친다.

```bash
python -m bench --config asr.qwen-seg-en+mt.qwen3.5-4b --dataset fleurs-en-ko
```

`configs/pipelines/asr.qwen-seg-en+mt.qwen3.5-4b.yml` — **재는 대상 그 자체다.** 무엇이 오디오를 넣어 주는지는
여기 없고, 그래서 이 파일은 bench 전용이 아니다. 이름은 부품을 역할별로 적는다 — ASR 은
`qwen-seg`(프로덕션 커밋 경로를 옮긴 Qwen3-ASR), 번역(MT)은 `qwen3.5`(Qwen3.5-4B). 시험용
`mock.yml` 도 있다.

**모델은 언어마다 다르다 — 프로덕션이 그렇다.** 서버 하나는 모델 하나이고 클라이언트가 언어에
맞는 서버를 고르므로, 파이프라인 설정도 언어마다 하나다. `asr.qwen-seg-en+mt.qwen3.5-4b` 는 영어
`<SEG>` 파인튜닝 `Doo12/Qwen3-ASR-1.7B-en-silence-c80-merged`, `asr.qwen-seg-ko+mt.qwen3.5-4b` 는
한국어 `Doo12/Qwen3-ASR-1.7B-ko-silence-v4c900-merged` 다. 언어가 섞인 `fleurs/ko_kr+en_us` 도
서버 하나가 듣는 상황 그대로 한국어 설정으로 잰다.

```yaml
pipeline:
  name: cascade
  transcription:
    name: qwen-seg
    model: Doo12/Qwen3-ASR-1.7B-en-silence-c80-merged
    chunk_size_sec: 2.0      # 모델별 설정은 그 모델 밑에 둔다
    max_new_tokens: 128
    dot_commit_confirm: true
    dot_commit_stall_chunks: 1
    rep_dedup: true
  translation:
    name: qwen3.5
    url: http://127.0.0.1:8100/v1
    model: Qwen/Qwen3.5-4B
    gpu_memory_utilization: 0.52
    context_turns: 1
  vad:
    name: silero
    min_silence_ms: 800
commit: seg                  # seg | punct | always
gpu_memory_utilization: 0.30
```

`configs/datasets/fleurs-en-ko.yml` — 데이터셋 하나와 목표 언어 하나다. 같은 데이터셋을 다른
목표 언어로 재려면 파일을 하나 더 둔다. 지금 있는 것은 `fleurs-en-ko`(`fleurs/en_us` → ko)와
`fleurs-ko-en`(`fleurs/ko_kr` → en)이다. 둘은 **같은 270문장**이다 — FLEURS test 에서 영어와
한국어 오디오가 모두 있는 문장이 270개이고, 위 설치 명령이 두 데이터셋을 서로를 번역 참조로
삼아 만들면 양쪽 모두 정확히 그 270개가 된다(다른 언어를 참조로 더하면 그 언어의 참조가 없는
문장이 빠져 둘이 어긋난다).

```yaml
dataset:
  name: fleurs/en_us   # $STITY_DATA_ROOT 아래 디렉토리. 여러 언어로 된 코퍼스는 언어마다 하나다
target: ko             # 들린 말을 모두 이 언어로 번역한다
```

**원문 언어는 설정에 없다 — 데이터셋이 정한다.** `dataset.yml` 의 `languages` 가 그 데이터셋에서
들릴 수 있는 언어이고, 보통 하나다(`fleurs/en_us` 는 `[en]`). 여러 언어가 섞인 입력을 재려면 그런
데이터셋 디렉토리를 따로 만든다 — `dataset.yml` 에 언어를 여럿 적고 항목마다 `src_lang` 을 두면
설정은 바뀌지 않는다.

데이터셋이 크면 `limit: N` 으로 앞의 N개만 돈다(`longform` 이면 발표 N개).

```yaml
dataset:
  name: covost2
  limit: 200
```

### 발표를 통째로 흘리기 (`longform`)

```yaml
dataset:
  name: fleurs/ko_kr+en_us
  longform: true
```

기본은 항목(문장)을 하나씩 흘린다. `longform: true` 면 같은 `group` 의 항목들을 발표 하나로
묶어 **그 오디오 파일을 처음부터 끝까지 한 세션으로** 흘린다. 항목들은 버려지지 않고 그
발표의 참조 분절이 된다 — 행의 `reference_segmentation`(문장마다 `offset`·`duration`·전사·번역)이
IWSLT 의 segmentation yaml 과 같은 것이다. 그래서 그룹의 항목이 모두 한 오디오 파일 안의
`offset` 구간이어야 한다(`fleurs/ko_kr+en_us` 의 대화가 그렇다). 아니면 시작 전에 죽는다.

행 하나가 발표 하나라 문장 단위 지표는 달라진다.

- `longyaal_ms`·`longyaal_ca_ms` 가 나온다. OmniSTEval 이 가설을 참조 문장에 재분절한 뒤
  `is_longform=True` 로 YAAL 을 낸다 — IWSLT 2026 이 지연 기준으로 쓰는 값이다.
- `bleu`·`comet` 은 같은 재분절 결과로 문장마다 한 쌍씩 낸다. COMET 의 원문은 그 문장의 참조 전사다.
- `laal`·`yaal` 은 빠진다. 발표 전체를 문장 하나로 보는 값이라 의미가 없다.
- `wer`·`cer`·`fsl`·`token_emission` 은 발표 전체에서 그대로 나온다. 정렬 시각은 문장
  `offset` 만큼 밀어 발표 시각으로 바꾼다.

**일부만 돌리려면 `limit: N`** 을 쓴다(위 참고). 결과 디렉토리 이름은 그대로라, 일부만 돈
결과가 전부 돈 결과를 덮어쓴다 — 두 결과는 비교가 안 되므로, 오래 남길 비교에는 쓰지 않는다.

### 소음과 공간 입히기 (`augment`)

데이터셋 설정에 `augment` 를 적으면 깨끗한 오디오에 장소 소음을 섞고 공간의 울림을 입혀서
흘린다. 데이터셋 리포는 건드리지 않는다 — 입히는 일은 bench(`bench/augment/`)가 한다.

```yaml
dataset:
  name: fleurs/en_us
target: ko
augment:
  room:
    place: hall             # 장소는 하나다
    level: [0.5, 1.0]       # 울린 소리의 몫. 0 이면 원음 그대로, 1 이면 전부 울린 소리
  noise:
    place: cafe
    level: [0.1, 0.4]       # 소음 크기 / 말소리 크기 (RMS). 0.5 면 소음이 말의 절반 크기
  volume:
    min_level_db: -23       # 최소 소리 크기 (RMS, dBFS). 이보다 작은 클립만 여기까지 키운다
```

**설정 하나에 장소는 하나다.** 장소를 여럿 섞으면 점수가 떨어져도 어느 장소 탓인지 가릴 수 없다.
장소마다 설정 파일을 따로 둔다.

- **숫자 값은 모두 범위 `[low, high]` 다.** 항목마다 그 안에서 고르게 뽑는다. `level: 0.3` 처럼
  한 값만 쓰면 `[0.3, 0.3]` 이다.
- 소음과 울림의 `level` 은 비율이고, 최소 음량 `min_level_db` 만 dB 다. 말소리는 보통 -26 ~ -20 dBFS 다.
- **무작위지만 항목마다 늘 같다.** 뽑기는 `데이터셋 이름/항목 id` 와 그 효과의 설정으로 정해진다.
  같은 설정으로 다시 돌리면 같은 소리가 나고, 효과 하나의 설정을 바꿔도 다른 효과의 선택은 그대로다.
- 순서는 울림 → 소음 → 음량이다. 녹음된 장소 소음에는 그 장소의 울림이 이미 들어 있어서
  소음을 울림 뒤에 섞는다. 음량은 섞인 결과 전체를 한 번에 키우므로 소음과 말의 비율은 그대로다.
- `volume` 은 바닥만 정한다. `min_level_db` 보다 작은 클립은 거기까지 키우고, 이미 큰 클립은
  건드리지 않는다. FLEURS 영어처럼 녹음이 아주 작은 데이터셋도 이 바닥 아래로는 들어가지 않는다.
  키우다가 소리가 깨질 만큼 커지면 거기서 멈추고, 실제로 키운 양이 `gain_db` 에 남는다(안 키웠으면 0).
- 울림은 직접음 위치에 맞춰 입혀서 말이 늦게 들리지 않고, 길이도 원래와 같다.
- 항목마다 무엇을 골랐는지(파일·시작 위치·level)가 행과 `item_open` 이벤트의 `augment` 에 남는다.
- 결과 디렉토리 이름은 데이터셋 설정 이름에서 온다. 소음 버전은 `fleurs-en-ko.cafe.yml` 처럼
  **파일을 따로 두어야** 깨끗한 실행을 덮어쓰지 않는다.

소음과 울림 파일은 `$STITY_DATA_ROOT/augment/` 아래에 장소 이름의 디렉토리로 둔다.
디렉토리 안의 `.wav` 는 모두 후보고 항목마다 하나를 고른다. 장소 이름은 곧 디렉토리 이름이라, 없는 장소를 적으면
있는 장소 목록을 알려주고 시작 전에 죽는다.

```
$STITY_DATA_ROOT/augment/
  noise/<place>/*.wav    # 예: DEMAND(cafe·station·traffic…), TAU Urban Acoustic Scenes(airport…)
  room/<place>/*.wav     # 공간의 임펄스 응답(RIR). 예: OpenAIR(hall·church), BUT ReverbDB(office)
```

`room` 은 소리가 거쳐 가는 곳이면 공간이 아니어도 된다. 전화기처럼 기기를 거친 소리도 그 기기의
임펄스 응답을 `room/phone/` 에 두면 같은 방식으로 입혀진다. 이때 `level` 은 1 로 둔다 — 원음이
섞이면 기기가 걸러낸 대역이 되살아난다.

지금 있는 설정은 `fleurs-ko-en.cafe`(카페 소음)와 `fleurs-ko-en.hall`(홀의 울림)이다. 둘 다
`volume` 으로 -23 dBFS 를 바닥으로 둔다.

**이름이 틀리면 있는 이름들을 알려주고 죽는다.** 설정 안의 오류도 그 설정 파일 기준 경로로
나온다 — `commit: unknown mode 'nope'`, `dataset: unknown key(s) ['nmae']`.

**부품은 파이프라인 안에 쓴다.** transcription·translation·vad 는 파이프라인의 인자이지
형제가 아니다. 다른 파이프라인이 쓰지 않는 부품을 받는 파이프라인도 자기 options 에 적으면
되고, 바깥이 그 종류를 먼저 알아야 할 일이 없다.

`translation: gpt` 처럼 문자열만 쓰면 `{name: gpt}` 로 풀린다.

**오디오를 어떻게 먹이는지는 설정이 아니다.** 오디오는 항상 실시간 속도로 먹이고, 청크
200ms 와 클립 뒤에 붙이는 침묵 4000ms 는 `__main__.py` 의 상수다. 이건 클라이언트를 기술하는 값이지 재는
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

### 한 실행은 한 청자다

`target` 은 들린 말을 모두 번역해 받을 언어 하나다. 서버에서 클라이언트마다 목표 언어를 하나
고르는 것과 같다 — 실행 하나가 청자 한 명이다. 양쪽이 서로의 언어로 받는 대화를 재려면 `target`
을 바꿔 두 번 돌린다.

어느 언어로 들렸는지는 파이프라인이 정한다. bench 는 데이터셋의 `languages` 와 `target` 만
넘기고, **항목의 실제 언어는 넘기지 않는다.** 넘기면 언어 판정을 정답으로 대신하게 되어 언어 판정
지표가 늘 1.0 근처로 나온다.

**목표 언어로 들린 말은 번역하지 않고 그대로 넘긴다** (`fleurs/ko_kr` 를 `target: ko` 로 돌리거나,
`[en, ko]` 가 섞인 데이터셋에서 한국어 발화). 그래서 번역 지표(BLEU·COMET·LAAL·YAAL·LongYAAL)는 원문 언어가 목표 언어와 같은 항목을
빼고 낸다 — `ko-ko` 쌍은 ASR 을 한 번 더 재는 것일 뿐이라 평균을 부풀린다. WER·CER 과 언어 판정은
모든 항목으로 낸다.

**언어 판정은 따로 잰다** (`lang_detect_accuracy`·`confusion`). ASR 이 커밋한 전사(`transcribed`)의
언어를 항목의 참조 언어와 비교한다. 번역기가 추정한 언어는 섞지 않는다. 판정이 틀리면 영어 발화를
한국어로 알고 번역 없이 넘기는 식의 오류가 나므로, 번역 지표와 별개로 봐야 한다.

**중간에 끊긴 발화**(항목의 `partial: true`, `fleurs/mix.py --cut` 이 만든다)는 전사가 실제로 말한
단어까지이고 번역 참조는 문장 전체의 것이다. 반쪽 문장의 번역 참조는 없으므로 번역 지표에서 빼고,
WER·언어 판정에는 쓴다.

**언어가 섞인 대화를 `longform: true` 로 흘리면** 발표 하나의 원문 언어가 `ko+en` 처럼 적히고
(WER 도 그 이름으로 묶인다), 문장마다의 언어는 `reference_segmentation` 에 남는다. 번역 지표는
재분절한 **뒤에** 목표 언어로 된 문장을 뺀다 — 먼저 빼면 그대로 넘긴 말이 옆 문장에 붙는다. 언어
판정은 커밋마다 그 커밋이 덮는 오디오와 가장 많이 겹치는 문장의 언어와 비교한다.

데이터셋의 `languages` 는 그대로 ASR 의 `allowed_languages` 가 되어 언어 이름 토큰에 로짓 바이어스를 건다.
즉 **언어 설정은 번역만이 아니라 WER 도 바꾼다.** 언어가 많을수록 제약이 약해진다.

거는지 마는지는 **파이프라인 쪽 설정**이다 — `transcription: {restrict_languages: false}`
로 끈다. 디코딩을 제약하는 값이라 재는 대상에 속하고, 어느 언어로 제약할지는 실행이
시작될 때 정해진다.

## 나오는 것

**설정 한 쌍이 디렉토리 하나다.** 이름은 `<파이프라인>-<데이터셋>` 이고 타임스탬프가 붙지
않는다. 같은 쌍을 다시 돌리면 그 디렉토리를 **덮어쓴다** — 그 쌍의 현재 답이 하나만
남는다는 뜻이다. 지우려면 그 디렉토리만 지우면 된다.

```
runs/asr.qwen-seg-en+mt.qwen3.5-4b-fleurs-en-ko/
  summary.json                      이 실행의 요약
  items.jsonl                       항목 단위 행. append (죽어도 채점된다)
  events.jsonl                      이벤트 전부. 리플레이가 읽는 것이다
  comet_inputs.jsonl                COMET 이 채점할 문장 쌍
  comet_scores.jsonl                그 문장마다의 COMET 점수 (같은 순서). 대시보드의 분포 그림이 읽는다
```

실행이 시작할 때 위 다섯 파일을 먼저 지운다(`reset_run_dir`). 스트림은 append 로 열리므로(중간에 죽어도
거기까지 채점된다) 안 지우면 지난 실행 뒤에 이어 붙어 두 실행이 한 기록으로 섞인다.

**설정은 여기 복사되지 않는다.** 입력은 `configs/` 에 있고 여기는 산출물만 둔다. 무엇을
돌렸는지는 `summary.json` 의 `config` 에 두 파일 내용이 그대로 들어가므로 따로 복사할
이유가 없다.

| 파일 | git |
|---|---|
| `runs/<name>/summary.json` | 추적한다 (작다, 실행 비교가 diff 로 보인다) |
| 나머지 | 안 한다 |

요약은 `items.jsonl` 을 다시 읽어 채점한 결과이고, 리플레이는 `events.jsonl` 을 읽는다. 둘 다
append 라 중간에 죽어도 그때까지가 남고 채점된다.

`summary.json` 의 `config` 는 설정 파일에 적힌 그대로이고, `components` 에는 설정에 안 쓴
값까지 **실제로 쓰인 값**이 들어간다. 두 실행을 비교할 때는 `components` 를 본다.

## 이벤트

한 줄 JSON 으로 흘린다(`core/utils/stream.py`). 시각 `t` 는 **그 항목의 시계를 켠 뒤 흐른
실제 초**이고, `audio` 는 그 순간까지 흘린 오디오 위치다. 오디오를 실시간으로 흘리므로 두 값의
차이가 "오디오보다 얼마나 밀렸나" 다. 파이프라인이 내는 시각(`committed_elapsed_sec`,
`translated_elapsed_sec`)도 같은 시계라 `fsl`·`laal_ca_ms` 검산이 이 시계에서 성립한다.

INFO 이상의 로그도 같은 줄기에 `log` 줄로 섞인다. 그래서 전사기가 커밋을 버린 이유
(`[DROP] rule=no-speech`, `rule=tail-filler` 처럼 규칙 이름이 붙는다)가 **그대로 보인다**.
번역이 0건인 항목의 이유를 여기서 찾는다.

빈 전사와 실패한 항목은 버리지 않는다. `summary.json` 의 `counts.empty_transcription_output` 로 세고 리플레이에 **항상**
넣는다(top-k 와 무관하게).

`counts.realtime_factor` 는 표준 RTF 다 — 파이프라인이 실제로 일한 시간(`counts.compute_sec`,
항목마다 `pipeline.listen`·`finish` 안에 있던 시간의 합) ÷ 먹인 오디오 길이(끝의 무음 포함).
1 보다 작으면 실시간보다 빠르다. 오디오를 실시간 속도로 먹이므로 벽시계(`wall_sec`)로는 속도를 잴 수 없다.

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
(`qwen3.py` 의 `_decode_chunks`·`_decode_tail` — 전사 백엔드가 모델을 부르는 자리. `qwen-seg` 도 물려받는다).
`_decode_stream`·`_decode` 는 모델이 실제로 돌 때만 이 둘로 넘긴다 — 청크가 2초 분량이
차기 전의 호출은 버퍼에 쌓기만 해서, 재면 200ms 마다 길이 0 짜리 막대가 생긴다.
스트리밍 중 디코딩은 `decode`, VAD·종료 커밋이 남은 꼬리를 푸는 디코딩은 `final_decode` 다.

```python
@timing.measure("final_decode")
async def _decode_tail(self, state) -> None:
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
돌려주는 기록(`Transcribed`·`Partial`·`Translated`)으로 다닌다. 그래서 `core/components` 의
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
| `fsl` | 커밋이 오디오보다 얼마나 늦게 도착했나 = `committed_elapsed_sec − decision_audio_sec`. **기록하지 않고 유도한다** — 두 시계가 이미 있으니 파이프라인이 따로 내면 어긋날 수 있다 |
| `laal` | OmniSTEval 의 LAAL(SimulEval 구현). 원문 언어가 목표 언어와 같은 항목은 뺀다. `decision_audio_sec` 이 `d_i` 이고, 소스 길이에서 자른다 — 끝에 붙인 무음 동안 커밋해도 발화보다 긴 지연이 나오지 않게. `laal_ca_ms` 는 벽시계 기준이고, 번역까지 끝나 전달된 시각(`Translated.translated_elapsed_sec`)을 `d_i` 로 쓴다 — 커밋 시각(`committed_elapsed_sec`)으로 재면 교정·번역 시간이 빠진다 |
| `bleu` | 원문 언어가 목표 언어와 같은 항목은 뺀다. **발화 하나가 한 쌍**이다 — 세그먼트를 다시 이어 붙여 채점한다. 아무것도 커밋하지 않은 발화는 빈 번역으로 채점한다. 언어 쌍마다 따로 채점하고(`bleu_by_pair`, 키는 `en-ko` 꼴) `bleu` 는 그 단순 평균이다 — 항목이 많은 쌍이 약한 쌍을 가리지 않게 |
| `comet` | `Unbabel/wmt22-comet-da`. 원문은 참조 전사, 번역·참조는 **BLEU 와 똑같은 문장 쌍**이다(`comet_inputs.jsonl`). 번역이 빈 문장은 모델에 넣지 않고 0점으로 센다 — 모델은 빈 번역에도 0.3~0.5 를 준다. Zouhar et al. 2024 ("Pitfalls and Outlooks in Using COMET", arXiv:2408.15366) 의 권고를 따른 것이다. 언어 쌍마다 평균을 내고(`comet_by_pair`) `comet` 은 그 단순 평균이다. **따로 도는 단계가 채점한다** — 아래 참고 |
| `yaal_ms` | OmniSTEval 의 YAAL. LAAL 과 분모가 같고, 소스가 끝나기 **전에** 나온 단위만 센다. 첫 단위가 소스 끝 이후에 나온 항목은 채점하지 않는다. `yaal_ca_ms` 는 벽시계 기준 |
| `longyaal_ms` | `longform` 실행에서만. OmniSTEval 이 가설을 참조 문장에 재분절(SoftSegmenter)하고 문장마다 YAAL 을 `is_longform=True` 로 낸 평균이다 — 문장 끝을 넘겨 나온 단위도 녹음이 끝날 때까지는 센다. 녹음 끝은 발표 오디오 길이다 — 마지막 참조 문장의 끝으로 잡으면 발화가 끝난 뒤 커밋되는 마지막 문장이 통째로 빠진다. `decision_audio_sec` 기준이고 벽시계 기준은 `longyaal_ca_ms`. 위 '발표를 통째로 흘리기' 참고 |
| `token_emission_ms` | 단어가 실제로 발화된 끝 시각부터 화면에 **바뀌지 않고 남은** 채로 처음 나타난 시각까지. 참조와 맞게 인식된 단어만 센다. `audio` 시계 기준이고, 벽시계 기준은 `token_emission_ca_ms`. 둘 다 평균·`_p50_ms`·`_p90_ms`. 발화 시각은 데이터셋의 `alignment.jsonl` 에서 온다 — 아래 참고 |
| `commit_reasons` | 사유별 비율. `finish` 비율이 크면 축의 커밋 경로가 안 도는 것이다 |
| `lang_detect_accuracy`·`confusion` | 언어 판정 정확도와 혼동 행렬. ASR 이 커밋한 전사의 언어를 참조 언어와 비교하고, 모든 항목으로 낸다. 판정 정확도는 참조 언어와 판정 언어를 둘 다 언어 코드로 맞춘 뒤 비교하고, 언어를 내지 않은 세그먼트는 `?` 로 틀린 것으로 센다 |

**COMET 은 별도 환경에서 돈다.** `unbabel-comet` 은 `protobuf<5`·`numpy<2` 를 고정하고 vLLM 0.14 는
`protobuf>=6.30` 을 요구해서, 한 환경에 둘을 같이 풀 수 있는 버전이 없다. 그래서 `bench/metrics/comet/` 이 자기 uv 환경을 갖고, `make bench` 가
`python -m bench` 프로세스가 끝난 뒤 그 환경에서 따로 부른다 — ASR 엔진과 번역 서버가 GPU 를 놓은 뒤라야
COMET 모델이 올라간다. 실행이 실패하면 부르지 않는다. 무엇을 채점할지는 bench 가 정한다 — `metrics/translation.py` 의 `translation_sentences`
가 BLEU 에 쓰는 문장 쌍을 원문과 함께 `comet_inputs.jsonl` 로 떨구고, `bench/metrics/comet` 은 그 파일만
읽어 한 번에 채점해 그 실행의 `summary.json` 에 `comet`·`comet_by_pair` 를 더하고, 문장마다의 점수를
`comet_scores.jsonl` 로 남긴다 — 항목 수만큼
도는 게 아니라 실행당 한 번이다. 그래서 항목 제외·재분절·괄호 속 이벤트 제거 규칙이 두 환경에 따로 있지 않다.
COMET 단계가 실패하면 `comet` 은 `unavailable` 에 그 이유로 남는다. 지난 실행을 다시
채점할 때는 그 한 줄만 부른다.

```bash
uv run --project bench/metrics/comet python -m bench.metrics.comet --run-dir bench/runs/asr.qwen-seg-en+mt.qwen3.5-4b-fleurs-en-ko
```

처음 한 번은 uv 가 환경(PyTorch 포함 수 GB)과 COMET 모델(약 2.3 GB)을 받는다.

**단어가 언제 발화됐는지는 데이터셋의 `alignment.jsonl` 에서 온다.** 데이터셋 리포의
`convert.py` 가 변환 끝에 강제정렬기로 만든다(그 리포 README 의 "정렬"). bench 는 파일이 있으면
읽고, 없으면 토큰 방출 지연만 빠진다. 전사가 바뀐 뒤 다시 정렬하지 않은 항목은 쓰지 않는다. 정렬은 재는
대상과 무관하다 — 참조 오디오와 참조 전사만 보므로 어떤 ASR 모델을 재든 같은 시각을 쓴다.
클립 길이를 정렬 간격(80ms) 넘게 벗어난 시각은 쓰지 않는다 — ACL6060 에서 19개 단어가 그랬다.

화면에 보인 시각은 `items.jsonl` 의 `records` 에 있다 — 파이프라인이 낸 레코드 전부가 낸 순서대로
`events.jsonl` 과 같은 줄 모양으로 들어 있고, 그중 `partial` 은 적힌 순간의 `t`·`audio` 를,
`transcribed` 는 자기 커밋 시각(`committed_elapsed_sec`·`decision_audio_sec`)을 쓴다. 화면 = 확정된 전사 + 지금의 partial 이고, 단어의
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
| 오디오만 (전사·번역 참조 없음) | `fsl`·`commit_reasons`·`lang_detect_accuracy` | `wer`·`cer`·`token_emission`·`bleu`·`comet`·`laal`·`yaal` |
| 전사는 있고 번역 참조 없음 | 위 + `wer`·`cer`·`token_emission` | `bleu`·`comet`·`laal`·`yaal` |

참조 번역이 없으면 `laal`·`yaal` 도 빠진다. 분모가 `max(|Y_hyp|, |Y_ref|)` 라서 `|Y_ref|` 를 빼면
근사가 아니라 **다른 지표(AL)** 가 되고, AL 은 짧게 생성할수록 점수가 좋아지는 구멍이 있다.
그걸 `laal_ms` 칸에 적으면 비교가 불가능한 두 숫자가 한 열에 섞인다.

## 구조

```
__main__.py   CLI + 실행. 파이프라인을 조립하는 유일한 모듈이다. 설정과 데이터셋을 먼저 읽고
              검사하므로 설정 오류는 모델을 올리기 전에 걸린다
config.py     데이터셋 설정을 읽고 파이프라인 설정과 합친다. 점수를 바꾸는 값 전부 명시
              해석. 파이프라인 설정 자체는 `core/config.py` 가 읽는다 — 서버와 공유한다
report.py     items.jsonl 행 + 채점 결과 → summary.json
replay/       events.jsonl·items.jsonl·summary.json → :9130 웹 페이지 둘
  runs.py           실행 목록, 데이터셋별 묶기, 대시보드가 쓰는 항목별 점수 분포
  session.py        실행 하나의 세션 재생 데이터 (항목·오디오·데이터셋 정보)
  server.py         라우팅: / 대시보드, /replay 세션 재생, /api/·/item/·/data/·/audio/
  static/           화면 전부: dashboard.html·.js, session.html, 둘이 함께 쓰는 common.css·.js
dataset.py    dataset.yml + manifest.jsonl (+ 있으면 alignment.jsonl) 읽기
augment/      데이터셋 설정의 augment → 흘리기 전 오디오에 울림·소음·음량을 입힌다
  config.py         설정 모양. 숫자는 모두 Range([low, high])
  augmenter.py      효과를 순서대로 부르고, 항목마다 같은 난수를 만든다
  assets.py         $STITY_DATA_ROOT/augment/<종류>/<장소>/*.wav 찾기와 읽기
  effects/          효과 하나가 파일 하나다. 순서는 __init__.py 의 ORDER (room → noise → volume)
metrics/      items.jsonl 행 → 지표. 모듈 하나가 지표 묶음 하나다
  score.py          score_row(항목별)·score_run(실행 전체)이 아래를 모두 부른다
  transcription.py  wer·cer
  translation.py    bleu, 그리고 BLEU·COMET 이 함께 쓰는 문장 쌍(translation_sentences)
  latency.py        fsl·laal·yaal·longyaal·token_emission
  asr.py            language_detection·commit_reasons
  common.py         여럿이 함께 쓰는 것: translated·per_lang·macro·정규화·재분절
  comet/            COMET. 자기 uv 환경에서 돈다 — __init__.py 가 없는 이유다. 있으면 그 환경에
                    없는 jiwer·omnisteval 을 import 하다 죽는다
```

채점은 `metrics/` 가 한다. 지표 하나가 함수 하나이고(`wer`·`cer`·`bleu`·`fsl`·`laal`·`yaal`·
`token_emission`·`language_detection`·`commit_reasons`), 모두 `items.jsonl` 의 행 목록을 받는다.
항목 하나의 값은 행 하나짜리 목록으로 부른다. `score_row` 가 항목별 값을, `score_run` 이 실행
전체 값을 모은다 — 설정에서 무엇을 계산할지 고르는 자리가 없으니 고를 코드도 없다.

`config.py`·`dataset.py`·`report.py`·`replay/` 는 `__main__.py` 를
import 하지 않는다.
그래야 GPU 없이 돌릴 수 있다.

데이터셋 리포는 반대로 **STiTy 를 import 하지 않는다.** 두 리포가 나란히 체크아웃돼
있다는 가정은 곧 깨진다.

## 아직 안 되는 것

- **동시 실행.** 항목은 한 번에 하나씩 순서대로 돈다.
- **WebSocket 경로와의 대조 검증(S6).** 데이터와 가중치가 있는 머신에서 해야 한다.
  같은 항목을 두 경로로 돌려 **전사 문자열과 커밋 사유 분포가 같은지** 보는 단계이고,
  이걸 통과하기 전에는 bench 숫자를 보고서에 쓰지 않는다.
