# core/

파이프라인 부품 + 번역 레이어 + 공용 채점 + 분절 연구 코드. 서버 코드는 포함하지 않는다.

| 경로 | 역할 |
|---|---|
| `translator/correct_and_trans.py` | `GPTTranslator` — 교정 + 번역 단일 GPT 호출. 서버 `--gpt-translation`으로 활성화 |
| `translator/gpt_corrector.py` | `GPTCorrector` — 교정만. 서버 `--correction`으로 활성화 |
| `translator/local_translator.py` | 로컬 번역기 — seq2seq(MADLAD/NLLB)와 `LLMTranslator`(지시형 LLM, **앞 발화를 문맥으로 받을 수 있는 유일한 백엔드**), 그리고 독립 번역 서버를 부르는 HTTP 클라이언트 `RemoteTranslator`. `make_translator` 가 모델 이름으로 고른다 |
| `translator/LOCAL_TRANSLATION.md` | 어떤 로컬 번역 모델을 올릴지, 문맥을 몇 턴 줄지 — 실측표 |
| `config.py` | **파이프라인 설정** — `configs/pipelines/*.yml` 하나를 읽어 부품까지 해석한다. 재는 대상(부품·커밋 정책·`gpu_memory_utilization`)만 적히고 무엇이 오디오를 넣어 주는지는 없으므로, bench 와 서버가 같은 파일을 읽는다. 이름으로 파일을 찾는 `resolve` 도 여기다 — `baseline` → `configs/pipelines/baseline.yml`. 데이터셋 설정은 그걸 가진 쪽(`bench/config.py`)이 읽는다 |
| `components/` | 파이프라인이 끼워 쓰는 부품. 종류별 서브패키지 하나, 그 안에 레지스트리 하나, 파일 하나가 백엔드 하나 — `vad/`(`detectors`) · `transcription/`(`transcribers`) · `translation/`(`translators`) · `correction/`(`correctors`). 설정 파일이 이름으로 고른다 |
| `pipeline/` | 부품을 엮는 쪽. `Pipeline` 과 `cascade`, 그리고 설정을 읽어 조립하는 `validate`·`build`·`describe`. **파이프라인은 부품이 아니다** — `Component` 를 상속하지 않고, 그래서 시간도 안 재진다 |
| `utils/metrics/` | 음성 번역 공용 채점 — WER·CER·BLEU·FSL·LAAL·커밋 사유·라우팅. 입구는 `score_item`·`score_run` 둘. STiTy `final` 필드를 그대로 읽고(`Utterance`/`Segment`), 데이터가 못 받치는 지표는 **값이 빠지고 `unavailable` 에 이유가 남는다** — `null` 도 예외도 없다. 실행·설정·보고서 형식은 모른다 |
| `meaning_segmentator/utils/` | 의미 분절 연구 스크립트 (GPT `<SEG>` 마킹, 점진적 컨텍스트 번역, COMET 평가) |
| `meaning_segmentator/autoseg/` | 분절 프롬프트 자동 생성 에이전트 루프. 코드가 하는 일 @meaning_segmentator/autoseg/AUTOSEG_SIMPLIFY.md, 사용법 @meaning_segmentator/autoseg/README.md |
| ⤷ 근거·기각 기록 | 왜 이 지표 조합인가, 무엇을 검토하고 버렸나, 순위 축 진단, 참조 기반 평가 프로토콜 @meaning_segmentator/autoseg/AUTOSEG_DETAILS.md |
| `meaning_segmentator/autoseg/baselines/` | Table 1a 타 정책 구현(`punct`/`syntax`/`causal_align`/`alignatt`/`mu_prefix`) + 강제정렬 타임스탬프 빌더 |
| `meaning_segmentator/docs/X2EN_DATASET.md` | {de,ja,zh}→en 트랙 데이터셋 구축 기록. 소스 언어별 단위·`T` 환산과 오염 방지 구간 배분 |
| ⤷ 문헌 대조 | @meaning_segmentator/docs/SEGMENTATION_CRITERIA_RELATED_WORK.md |
| `research/cif`, `research/context_scoring` | CIF·컨텍스트 스코어링 실험. 런타임 경로 아님 |

## `components/` — 부품 하나는 클래스 하나

부품을 쓴다는 것은 클래스 하나를 쓴다는 뜻이다. 이름, 받을 설정(`SETTINGS`), 그리고
자기가 하는 일 하나. 그 외에 구현해야 하는 것은 없다.

```python
@translators.register("my-translator")
class MyTranslator(Translator):
    SETTINGS = {"model": ("model", str)}

    async def load(self):
        self.model = ...

    async def translate(self, text, target_lang, source_lang=None, context=None):
        return translated_text, source_lang
```

| 종류 | 레지스트리 | 하는 일 |
|---|---|---|
| `transcription` | `transcribers` | `start(language)` / `transcribe(audio)` / `flush(reason, speech)` / `finish(reason, speech)` → 만든 `Transcribed`·`Partial` 목록 |
| `translation` | `translators` | `translate(text, target_lang, source_lang, context)` → (번역문, 소스 언어) |
| `vad` | `detectors` | `detect(audio)` → 끝난 발화 `Speech` 또는 `None` |
| `correction` | `correctors` | `correct(text, language)` → 고친 텍스트. **등록된 백엔드가 아직 없다** |
| `pipeline` | `pipelines` | `start` / `listen(audio)` / `finish`. `REQUIRED`·`OPTIONAL` 로 자기 부품을 선언한다 |

**백엔드를 등록하는 자리는 파일 그 자체다.** 종류 패키지의 `__init__.py` 는 레지스트리를
만들고 `discover(__name__)` 를 부를 뿐이고, 그 디렉토리의 모듈을 전부 import 한다 — 새
백엔드는 파일 하나를 떨구면 끝이고 지우면 사라진다. 목록을 손으로 적던 시절에는 그 import
줄이 재수출처럼 보여서, 안 쓰는 import 로 오해해 지우면 `unknown transcription: 'qwen3'`
으로 엉뚱한 데서 터졌다.

그 대가로 **모듈 최상단은 가벼워야 한다.** 전부 import 되므로 `torch`·`vllm` 같은 것을
파일 머리에서 가져오면 안 쓰는 실행까지 그 값을 치른다. 무거운 것은 지금처럼 `load()`
안에서 가져온다 — 종전에는 관례였고 이제는 요건이다.

부품은 파이프라인의 **옵션 안에** 설정한다. 형제 블록이 아니다 — 다른 파이프라인에 없는
부품이 필요하면 그 레지스트리를 `REQUIRED`/`OPTIONAL` 에 적는 것만으로 설정 키가 생긴다.
파이프라인이 안 쓰는 부품을 설정하면 오류다. 안 쓰는 부품은 블록을 **빼서** 없앤다 —
아무것도 안 하는 `none` 백엔드는 두지 않는다.

**부품은 만든 것을 반환한다.** 전사기는 `Transcribed`(되돌리지 않을 텍스트)와
`Partial`(지금 멈추면 이렇게 말하겠다는 것)을 만든 순서대로 돌려주고, 파이프라인이
번역을 붙여 `Final` 로 바꾼다. 세 기록은 `registry.py` 에 있는 frozen dataclass 라
필드가 닫혀 있다 — 오타가 조용히 빈칸이 되지 않는다.

**기록은 실행을 모는 쪽이 한 곳에서 한다.** `core/components` 와 `core/pipeline` 의 어떤 파일도
`core.utils.stream` 을 import 하지 않는다. 돌려받은 그 객체를 그대로 `events.jsonl` 에
쓰고 그대로 채점하므로, 기록에 남은 것과 채점되는 것이 **같은 것일 수밖에 없다**.
줄마다 `audio`(녹음 안에서의 위치)가 찍히므로 실시간보다 빠르게 민 실행도 말한 속도
그대로 재생된다.

**시간은 `registry.register` 가 자동으로 잰다.** 등록될 때 그 종류의 베이스가 선언한
메서드(`transcribe`·`flush`·`finish` / `translate` / `detect` / `correct`)에
`timing.measure` 를 씌우므로 **부품 코드에 계측이 한 줄도 없다** — 새 백엔드는 레인이
그냥 생기고, 새 종류는 그 베이스가 선언한 메서드가 곧 레인이 된다. 나가는 줄은
`{"t": 2.001, "type": "timing", "tag": "transcribe", "audio": 2.0, "dur": 0.353}` 이고
`t` 는 다른 줄과 같이 시작 시각이다. 읽는 쪽은 태그 목록 없이 규칙만 안다 — `dur` 이
있으면 시간이 걸린 구간, `tag` 가 레인 이름. **시작한 뒤 끝내지 않을 방법이 없다** —
데코레이터라 예외로 빠져나가도 막대가 닫힌다.

`load`·`close`·`start` 는 항목 시계 밖이라 안 잰다. 파이프라인도 안 잰다 — 부품을 엮는
쪽이고 그 막대는 부품 막대를 전부 덮는다. 메서드보다 잘게 재야 할 때만 `@timing.measure`
를 직접 붙인다(트리에 한 군데, `qwen_seg.py` 의 `_decode`). **태그는 모델이 아니라 하는
일이다** — `qwen3_generate` 가 아니라 `decode` 여야 다른 백엔드의 같은 구간과 한 레인에서
비교된다. 자세한 것은 [bench/README.md](../bench/README.md) 의 '시간은 자동으로 재진다'.

**로그는 `log.debug`·`log.info`·`log.warning`·`log.error` 넷뿐이고, 줄은 `[TAG] 문장`
으로 쓴다.** 태그는 스트림에 나갈 때 `tag` 필드로 떨어져 나가므로 산문을 파싱할 일이
없다. **숫자를 문장에 녹이지 않는다** — 그림에 찍히거나 채점되는 값은 로그가 아니라
반환값이다.

**스트리밍 전사는 이미 낸 텍스트를 고쳐 쓴다.** Qwen3 는 청크마다 `unfixed_token_num` 만큼
되돌려 다시 디코딩하고, 반복 환각 컷은 `state.text` 를 줄이기까지 한다. 커밋한 지점을
넘어 고쳐지면 `[REVISED]`(다시 씀) 또는 `[RETRACTED]`(거둬들임)이 로그에 남는다 —
무수정 제약상 되돌릴 수 없으니, 조용히 어긋나는 대신 기록에 남긴다. `qwen3` 는 그
기록만 남기고 지나간다.

**전사 백엔드는 둘이다.** `qwen3` 는 커밋 트리거(SEG·마침표·매 청크)만 그대로 옮긴
최소 구현이고, `qwen-seg` 는 프로덕션 서버(`Qwen3-ASR/examples/streaming_websocket_server.py`)
의 커밋 경로를 그대로 옮긴 것이다 — 커서 추적, 재방출 가드(`cross-dedup`·`cross-dedup-fuzzy`·
`committed-suffix-dedup` 등), dot 확정 게이트 네 규칙(문맥·합의·정체·종료), SEG/헤더/길이
초과 시의 슬롯 리셋과 오디오 carry, 꼬리 무음 트림과 짧은 발화 재시도, 언어 헤더·무음
정형문·꼬리 조각 폐기(`[HALLUC-DROP]`·`[TAIL-DROP]` 등 같은 태그로 로그에 남는다).
프로덕션과 같은 숫자를 재려면 이쪽이고, 그 방어가 없을 때의 바닥선을 보려면 `qwen3` 다.
`dot_commit_confirm`·`dot_commit_stall_chunks`·`rep_dedup` 은 `qwen-seg` 의 설정이다.

**서버와 어긋나면 벤치가 무엇을 재는지 알 수 없다.** 그래서 정답을 적어 두는 대신 양쪽에
같은 디코딩 대본을 먹여 커밋 목록이 같은지만 보는 대조 테스트를 둔다 —
`Qwen3-ASR/tests/test_qwen_seg_parity.py`. 서버를 고치면 이 테스트가 `qwen_seg.py` 도 같이
고치라고 알려 준다.

**무음 위 커밋 폐기(`[SILENCE-DROP]`)는 VAD 구간이 있어야 판정된다.** 서버가 보는 것은
"커밋 시점이 침묵인가"가 아니라 "이 커밋이 덮는 구간(직전 final 의 끝 ~ 이번 커밋)에 음성이
있었나"라, 진행 중인 구간까지 포함한 구간 목록이 필요하다. 파이프라인이 `start` 에서
`vad=` 로 검출기를 넘기고 전사 부품이 `detector.spans` 를 읽는다 — 검출기가 없거나 꺼진
실행에서는 판정 근거가 없으므로 이 가드가 통째로 비활성이다(서버의 `vad_enabled` 와 같다).

두 GPT 모듈 모두 기본 모델은 `gpt-5.4-mini` (런타임 경로. `autoseg/`의 모델과 무관하다). 두 플래그 모두 꺼져 있으면 서버는 Google Translate로 번역하므로 `core/`의 GPT 경로를 아예 타지 않는다.

## 규칙

- 학습/실험 코드는 `research/` 아래에만. 런타임 파일 옆에 두지 말 것.
- **주석을 쓰지 않는다.** `components/`·`pipeline/` 과 `utils/metrics/` 에는 주석도 독스트링도 없다.
  주석으로만 알 수 있는 것이 있으면 그건 코드가 잘못된 것이다 — 이름과 구조로 드러내고,
  경위는 커밋 메시지에 적는다.
- **문맥은 LLM 백엔드만 받는다.** 서버의 `--local-translation-context N` 이 앞 발화 원문을 넘기고,
  seq2seq 번역기는 그걸 받아서 버린다(경고 1회). 이어붙여 넣는 `--google-context` 방식은 Google 이
  줄바꿈을 보존해 주기 때문에 되는 것이라 로컬 모델에서는 깨진다 — NLLB 는 줄 수를 안 지켜
  문맥 덩어리가 통째로 자막에 나가고, MADLAD 는 번역 대신 잡음을 뱉는다. 실측과 문맥 깊이별
  수치는 [translator/LOCAL_TRANSLATION.md](translator/LOCAL_TRANSLATION.md).
- 런타임 경로(`translator/`) 의존성은 GPT 쪽이 stdlib + `openai`, 로컬 번역기가 `transformers`/`torch` 다. 둘은 서로를 요구하지 않으며, `translator/__init__.py` 가 이름을 쓸 때 가져오는 것도 그래서다 — 로컬 번역 서버는 `openai` 없이 뜬다. `bitsandbytes` 는 `LLMTranslator` 를 4bit/8bit 로 올릴 때만 더 필요하고, 없으면 `--quant none` 으로 뜬다. 연구 스크립트는 각자 `requirements.txt`를 갖는다 (예: `meaning_segmentator/requirements.txt`).
- `autoseg/`는 Letsur AI Gateway / OpenAI / OpenAI 호환 로컬 서버를 쓴다 — 상대는 **`--provider {letsur,openai,local}`** 가 정하고(기본 `letsur`), 키는 그 프로바이더의 환경변수 하나만 본다 (`letsur`→`LETSUR_API_KEY`, `openai`→`OPENAI_API_KEY`, `local`→키 불필요). 환경변수를 순서대로 뒤지거나 `sk-` 접두사로 추측하던 종전 방식은 없앴다 — `.env` 에 키가 둘이면 명령줄만 봐서 어디로 갔는지 알 수 없었고 런 기록에도 안 남았다. 프로바이더와 해석된 `api_base_url` 은 런의 `config.json` 에 기록된다. 기본 모델은 `gpt-5-mini` (실측 근거 @meaning_segmentator/autoseg/AUTOSEG_DETAILS.md '순위 축 진단'). 에이전트 호출은 `httpx`만 있으면 되지만, 지표 백엔드(COMET/CometKiwi)는 `unbabel-comet` + GPU가 필요하다. CometKiwi는 HF 게이트 모델이라 라이선스 동의 + `hf auth login` 선행 (구버전은 `huggingface-cli login`).
- **관문을 먼저 통과시킨다.** consistency 백엔드를 바꾸면 `validity_check.py`, adequacy 백엔드를 바꾸면 `adequacy_check.py`, 판정자 모델이나 `JUDGE_SYSTEM`을 바꾸면 `judge_check.py`. 지표는 틀리면 숫자로 드러나지만 판정자는 조용히 루프를 발산시킨다. NLI contradiction 의 잡음 바닥·순위 정렬 재검은 `noise_floor.py`.
- 언어별 자원(형태소 분석기·의존 파서)을 `autoseg/`에 넣지 말 것. 언어 지식은 `measured_profile.json`(측정)과 `language_profile.json`(LLM)으로만 들어간다.
