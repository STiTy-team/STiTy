# 스트리밍 ASR 백엔드 다변화

| | |
|---|---|
| 브랜치 | `feat/multi-asr-backends` (base `origin/main`) |
| 실행 호스트 | RTX 4090 · 24564 MiB · driver 580.95.05 |
| 워크트리 | `~/bench-wt/multi-asr-backends` (산출물은 워크트리 밖 `~/bench-results/`) |
| 런 디렉터리 | `20260909_204118` · `qwen3final_20260909_230132` · `nemogap_20260909_232203` |
| 상태 | **완료** — 세 백엔드 전부 전사 성공, GPU 반납 확인 |
| 작성 / 갱신 | 2026-09-07 / 2026-09-10 |

Qwen3-ASR 단일 백엔드 체제에 Voxtral Realtime과 Nemotron Streaming을 추가한다.
후보 6종 심사, 측정 방법론 확정, 백엔드 구현, 실측까지 마쳤다. 결과와 그 과정에서 확정된
함정은 8~10절에 있다.

---

## 1. 배경

STiTy의 ASR 경로는 Qwen3-ASR 하나다. "현재 스트리밍 ASR 기술 수준이 어디까지 왔는가"에
답하려면 비교군이 필요하고, 그 비교군은 **한국어를 하면서 실제로 스트리밍인** 모델이어야
한다. 이 두 조건이 후보 심사의 축이 됐다.

초기 후보는 NVIDIA Parakeet-TDT, NVIDIA Nemotron streaming, Mistral Voxtral Realtime,
Kyutai STT, Moonshine 다섯이었다. 심사 결과 둘을 채택하고 하나를 제외했으며, 목록에 없던
한 모델이 사실상 최적 후보로 올라왔다.

---

## 2. 후보 심사

| 모델 | 라이선스 | 특허 그랜트 | 네이티브 스트리밍 | 한국어 | 판정 |
|---|---|---|---|---|---|
| **Voxtral Mini 4B Realtime** (Mistral, 13개 언어) | Apache-2.0 | 있음 | end-to-end 학습 | 지원 | **채택** |
| **Nemotron 3.5 ASR Streaming** (NVIDIA, 600M, 40 locale) | OpenMDW-1.1 | 있음 | cache-aware FastConformer | 지원 (`ko-KR`) | **채택** |
| Kyutai STT 1B / 2.6B | CC-BY-4.0 | 없음 | delayed streams | en / fr 전용 | 영어 참조 |
| Parakeet-TDT 0.6B v3 | CC-BY-4.0 | 없음 | 오프라인 | 유럽어 25종 | 속도 상한 참조 |
| Moonshine | 영어 MIT / 비영어 NC | — | 오프라인 | 비상용 한정 | **제외** |
| Qwen3-ASR 1.7B *(현행)* | Apache-2.0 | 있음 | 청킹 + 커밋 정책 | 지원 | 기준선 |

---

## 3. 핵심 발견

### F1 — 초기 후보 다섯 중 셋이 한국어를 못 한다

Parakeet v3는 유럽어 25종, Kyutai는 영어·프랑스어뿐이다. Moonshine의 한국어 변종은
존재하지만 비상용 라이선스다. STiTy가 한↔영 동시통역인 이상 이 셋은 영어 쪽 baseline
이상의 역할을 할 수 없다.

### F2 — 후보 다섯 중 둘은 애초에 스트리밍이 아니다

Parakeet-TDT와 Moonshine은 오프라인 모델이다. 청킹으로 감싸 스트리밍 비교에 넣으면
측정되는 건 모델이 아니라 **우리 커밋 정책**이다. Qwen3-ASR이 이미 그 구조이므로,
같은 교란요인을 하나 더 늘리는 셈이 된다.

### F3 — 목록에 없던 Voxtral Realtime이 최적 후보였다

Apache-2.0이고 13개 언어가 한국어·일본어·중국어·스페인어를 포함한다 — STiTy 지원
언어셋과 정확히 겹친다. 지연을 재학습 없이 80~2400ms에서 조절할 수 있고, Qwen3-ASR과
같은 vLLM 스택으로 서빙된다.

### F4 — 라이선스 통념이 반대였다: CC-BY-4.0에는 특허 그랜트가 없다

초기 검토에서 OpenMDW-1.1(Nemotron)을 "법무 확인 필요"로 표시했으나 이는 틀렸다.
OpenMDW-1.1은 **copyright + patent + database + trade secret** 에 걸쳐 무제한 권리를
부여하며, field-of-use·로열티·지역 제한이 없고 출력물에도 제약이 없다. 재배포 시 고지
유지가 유일한 의무다.

반면 Creative Commons 라이선스는 특허권을 부여하지 않는다고 본문에 명시돼 있다.
상업 배포 관점에서 확인이 필요한 쪽은 오히려 CC-BY-4.0 모델들(Parakeet, Kyutai)이다.
다만 둘 다 한국어 미지원이라 실무 영향은 없다.

---

## 4. 측정 방법론

### 단일 WER로는 비교가 성립하지 않는다

모델마다 지연 노브의 **단위가 다르다.**

| 모델 | 노브 | 단위 |
|---|---|---|
| Voxtral | `transcription_delay_ms` (`tekken.json`) | 밀리초 |
| Nemotron | `att_context_size = [56, R]` | 80ms 프레임 개수 |
| Qwen3-ASR | 청크 길이 + 커밋 정책 | 초 |

각자의 기본값으로 돌려 WER을 비교하면 그 차이가 *모델 성능 차이인지 지연 예산 차이인지
구분되지 않는다.*

### 실제 설정 가능한 지연 지점

| 지연 (ms) | 80 | 160 | 320 | 480 | 500 | 560 | 1120 | 1200 | 2400 | 2500 |
|---|---|---|---|---|---|---|---|---|---|---|
| Voxtral | ● | ● | ● | **◉ 기본** | | ● | ● | ● | ● | |
| Nemotron | ● | ● | ● | | | ● | **◉ 기본** | | | |
| Kyutai 1B | | | | | ● 고정 | | | | | |
| Kyutai 2.6B | | | | | | | | | | ● 고정 |
| Qwen3-ASR | *ms 노브 없음 — 초 단위 청크 + 커밋 정책이 지연을 정한다* | | | | | | | | | |

Voxtral은 80ms 배수 `[80, 1200]` 구간 + `2400`, Nemotron은 `R ∈ {0,1,3,6,13}` 에
대응하는 5점뿐이다. **겹치는 값이 거의 없다는 것이 요점이다** — 기본값끼리 비교하면
Voxtral 480ms 대 Nemotron 1120ms로, 지연 예산을 2.3배 다르게 준 채 품질을 견주게 된다.

따라서 **지연을 맞추고 품질을 재거나, 품질을 맞추고 지연을 잰다.** 어느 쪽이든 맞출
지점을 찾기 위한 최소한의 곡선이 필요하다.

> **선례**
> `arabic` 브랜치의 커밋 `4ef4c47` 은 "아랍어 static 스윕 6~9초 — 8초에서 평탄,
> **이전 결론 정정**" 이다. 한 점만 봤을 때는 "지연을 더 주면 좋아진다"였으나 곡선을
> 채우자 평탄 구간이 드러났다. 무릎점은 점 하나로는 보이지 않는다.

다만 전 데이터셋 풀 스윕은 낭비다. 채택한 절차는 **서브셋에 모델당 3점 → 지연을 맞춘
공통 동작점 확정 → 그 한 점에서만 전체 평가** 다. 스윕은 배포 방식이 아니라 동작점을
고르기 위한 일회성 측정이다.

---

## 5. 기술 제약

### 의존성이 공존 불가다

| 환경 | 핵심 버전 | 비고 |
|---|---|---|
| `stity` *(기존)* | torch 2.9.1+cu128 · transformers 4.57.6 · vllm 0.14.0 | Qwen3-ASR |
| `asr-nemotron` | transformers ≥ 5.13.0 | 4.57.6과 충돌 |
| `asr-voxtral` | vllm ≥ 0.20.0 · mistral-common[audio] ≥ 1.9.0 | 0.14.0과 충돌 |

세 모델을 한 env에 넣는 것은 불가능하다. 그러나 평가 하네스가 **WebSocket으로만** 서버에
붙기 때문에 이는 장애물이 아니라 전제가 된다 — 서버가 어느 env에서 돌든 무관하다.
env 분리가 정석이다.

### 레포에 오디오가 없다

평가 데이터는 git에 추적되지 않는다. 실측 결과 사용 가능한 것은 아래뿐이며,
**LibriSpeech test-other는 존재하지 않는다.** 영어 트랙은 FLEURS `en_us` 로 대체했다.

| 데이터 | 경로 | 클립 | 용도 |
|---|---|---|---|
| FLEURS ko_kr | `~/STiTy-team/datasets/fleurs/data/ko_kr` | 382 | CER · 스모크 + 스윕 |
| FLEURS en_us | `~/STiTy-team/datasets/fleurs/data/en_us` | 647 | WER · 스모크 |
| KsponSpeech | `evaluation/KsponSpeech/sample_data` | 40 | 예비 |
| ~~LibriSpeech test-other~~ | `evaluation/LibriSpeech/LibriSpeech/` | 0 | 부재 |

---

## 6. 통합 지점

프로덕션 서버는 3198줄이지만 ASR 엔진에 닿는 지점은 셋뿐이다.

```
Qwen3ASRStreamingServer.init_model()                  # :2756  모델 1회 로드
Qwen3ASRStreamingHandler._asr_streaming_transcribe()  # :1077  청크 → 가설 텍스트
Qwen3ASRStreamingHandler._asr_finish_streaming()      # :1308  종료 플러시
```

그리고 FSL 평가 서버가 **정확히 뒤의 두 메서드만 오버라이드**한다
(`streaming_websocket_server_fsl.py:117`, `:178`). VAD·커밋 정책·중복 가드·번역·타이밍
계측은 전부 백엔드 무관이다.

초기 검토에서 `core/modules.py` 의 `SpeechRecognizer` 추상 클래스를 채우자고 제안했으나
철회했다. 그 파일은 `origin/main` 에 존재하지 않으며 `arabic` 브랜치에만 남은 잔재다.

> **프로토콜 함정**
> `finish_done` ack는 선택이 아니다. 하네스의 `POST_FINISH_GRACE_SEC` 가 60초이므로,
> 이 ack를 보내지 않는 서버는 **파일마다 60초를 통째로 대기**하게 만든다.

---

## 7. 구현물

`evaluation/backends/`

| 파일 | 역할 |
|---|---|
| `protocol.py` | 백엔드 중립 WS 서버. vllm/transformers를 import하지 않는다 — env 분리가 성립하는 이유. |
| `engine_nemotron.py` | cache-aware FastConformer-RNNT 엔진. `att_context_size` 로 지연 조절. probe 모드 내장. |
| `probe_voxtral.py` | vLLM `/v1/realtime` 스키마 수집기. `/openapi.json` 과 이벤트를 그대로 덤프한다. |
| `smoke_client.py` | FLEURS ko/en 클라이언트. 외부 의존성 없이 편집거리로 CER/WER 자체 채점. |
| `server.py` | 백엔드 선택 진입점. 엔진 모듈을 지연 import한다. |
| `smoke_voxtral.py` | Voxtral 전용 클라이언트. `smoke_client.py` 의 채점·데이터 로더를 그대로 써서 요약 JSON 형식을 맞춘다. |
| `run_backend_bench.sh` | 무인 드라이버 v1. 백엔드 하나가 죽어도 계속 가고, `trap EXIT` 로 GPU를 반드시 반납한다. |
| `asr_bench_driver.sh` | 무인 드라이버 v3. 실제로 결과를 낸 쪽. conda 절대경로 + Qwen3 기준선 인자(9절 R3)가 들어 있다. |
| `run_voxtral.sh` | Voxtral vLLM 기동. `--max-model-len 16384` 와 `VLLM_USE_FLASHINFER_SAMPLER=0` 이 필수다(9절 R2). |
| `qwen3_fix_step.sh` | Qwen3 기준선 단독 재실행. `--gpu-memory-utilization` 과 서버 인자를 밖에서 주입해 R3 를 좁히는 데 쓴 스텝. |
| `nemo_gap_step.sh` | Nemotron 스윕 공백(rc0 ko · rc6 en) 보충. 케이스마다 서버를 내려 GPU 를 반납하고 다음 케이스로 간다. |

### 범위에서 제외한 것과 근거

- **엔진 인터페이스 리팩터** — 동작 보존 리팩터는 회귀 게이트가 있어야 의미가 있는데,
  미검증 백엔드 둘과 함께 무인으로 돌리면 숫자가 틀어졌을 때 원인을 귀속시킬 수 없다.
- **전 데이터셋 풀 평가** — 미검증 어댑터로 몇 시간짜리 풀런을 돌리면 실패 시 그 시간이
  전부 낭비된다.
- **Voxtral 브리지** — `/v1/realtime` 이벤트 스키마를 실행해본 적이 없다. 눈 감고 브리지를
  쓰는 대신 프로브로 실제 스키마를 확보한 뒤 왕복 한 번에 정확히 짠다.

---

## 8. 실행 결과

세 백엔드 모두 실제로 전사에 성공했다. FLEURS, 실시간 스트리밍, 4090 단독 점유 조건.

### 3백엔드 비교

| 백엔드 | ko CER | en WER | FSL ko / en (s) | RTF ko / en | 빈 전사 | 클립 |
|---|---|---|---|---|---|---|
| **Qwen3-ASR 1.7B** *(기준선)* | **0.0254** | **0.0357** | 0.375 / 0.266 | 1.12 / 1.13 | 0 | 20 / 20 |
| Voxtral Mini 4B Realtime | 0.0556 | 0.1626 | 2.333 / 1.902 | 1.49 / 1.86 | 2 (en) | 20 / 20 |
| Nemotron 0.6B @ rc6(560ms) | 0.0932 | 0.1026 | **0.037 / 0.043** | 1.10 / 1.13 | 0 | 50 / 50 |

Nemotron 은 각 언어의 최적 lookahead 기준이다. 초기 표의 rc13 수치(ko 0.0999 / en 0.0992)에서
ko 0.0932(50클립), en 0.1026(50클립)으로 정정했다.

> **Voxtral 의 지연 수치는 다른 둘과 같은 뜻이 아니다.**
> 이 배포는 첫 `transcription.delta` 가 오디오를 다 보낸 뒤에 온다(첫 토큰 지연 ≈ 오디오 길이).
> 증분 스트리밍이 아니라 **commit 트리거**로 동작하므로, FSL·FTL 을 Qwen3·Nemotron 과
> 직접 비교하면 안 된다. 정확도(CER/WER)만 같은 축에서 읽을 수 있다.

### Nemotron 지연-정확도 곡선 (FLEURS ko, 케이스당 50클립)

| lookahead | 지연 | ko CER | FSL (s) | RTF |
|---|---|---|---|---|
| rc0 | 80ms | 0.1191 | 0.577 | 1.253 |
| rc3 | 320ms | 0.1049 | 0.037 | 1.111 |
| **rc6** | **560ms** | **0.0932** | 0.037 | 1.100 |
| rc13 | 1120ms | 0.0960 | 0.043 | 1.103 |

**rc0 은 완전히 지배당한다(dominated).** 정확도·FSL·RTF 셋 다 최악이고 FSL 은 rc6 의 15배다.
lookahead 를 줄여 지연을 얻는 트레이드오프가 성립하지 않는다 — 청크를 잘게 쪼개면서 늘어난
호출 오버헤드가 lookahead 절감분을 통째로 먹는다. **80ms 는 쓸 이유가 없고, 560ms 가 명확한
무릎점**이다. 4절에서 예고한 "점 하나로는 무릎점이 안 보인다"가 그대로 재현됐다.

en 은 lookahead 에 둔감하다(rc6 0.1026 / 50클립 vs rc13 0.0992 / 20클립 — 표본이 달라 직접
비교는 불가하나 둘 다 ~10%).

---

## 9. 근본원인 3건

### R1 — Nemotron 의 정식 API 는 청크별 `generate()` 가 아니다

가장 큰 함정. 청크마다 `model.generate()` 를 부르면 **매 호출이 encoder/decoder 캐시를 리셋**해
청크 경계의 단어가 통째로 빠진다. 예외도 경고도 없이 조용히 열화되므로 전사를 눈으로 보지 않으면
못 잡는다. 같은 문장의 실측 비교:

```
청크별 generate : 매입국 충격 신혼단계가 때문에 문화충격보다 빠르게 발생하 더 각심할 수
제너레이터 API  : 제입국 충격은 신혼 단계가 없기 때문에 문화충격보다 빠르게 발생하며
                  더 오래 지속하고 더 극심할 수 있습니다.
```

정답은 **mel 청크를 내놓는 제너레이터를 `input_features` 로 한 번 넘기는 것**이다
(`transformers/models/nemotron_asr_streaming/generation_nemotron_asr_streaming.py:212-225`).
`num_lookahead_tokens` 도 같이 넘겨야 한다. 푸시 기반 WS 서버에서는 큐 백드 제너레이터와
워커 스레드로 감쌌다(`engine_nemotron.py`).

부수 사실 넷:

- **청크 크기는 샘플이 아니라 mel 프레임 수로 고정된다.** `first = 1 + 8*right`,
  `subsequent = 8*(right+1)`. lookahead 13 이면 105 / 112.
- 프로세서의 `num_samples_first_audio_chunk` 는 **첫 청크에서 1프레임 과다**다(16840 → 106,
  필요한 값은 105). 프레임 수에서 역산할 것: `(need-1)*160`. `num_samples_per_audio_chunk` 는 정확하다.
- 노브는 4절에 적은 `att_context_size` 가 아니라 **`num_lookahead_tokens`** 이고 `0/3/6/13` 만
  유효하다(각 80/320/560/1120ms). `1` 같은 값을 주면 **조용히 13 으로 떨어져** 다른 이름의 같은
  결과가 저장된다 — 스윕에서 특히 위험하다.
- `generate()` 는 텐서가 아니라 `NemotronAsrStreamingGenerateOutput` 을 반환한다 → `out.sequences` 를 디코드.

### R2 — Voxtral 은 벽이 셋이었다

1. **KV cache 예산.** `--gpu-memory-utilization 0.60` + 기본 `max_model_len 131072` 조합은
   **-5.25GiB**로 기동 자체가 불가능하다. `0.85` + `--max-model-len 16384` → +9.87GiB.
2. **이 박스에 nvcc 가 없다.** flashinfer 의 top-k/top-p 샘플러가 JIT 로 CUDA 커널을 빌드하려다
   EngineCore 를 죽인다. 어텐션·컴파일 문제가 아니므로 **`VLLM_USE_FLASHINFER_SAMPLER=0` 만**
   끄면 되고 PIECEWISE 컴파일과 CUDA 그래프는 정상 동작한다. `--enforce-eager` 로는 안 풀린다.
3. **Realtime 프로토콜.** `session.update` 의 `model` 은 **최상위 필드**여야 한다. `session` 안에
   넣으면 `Missing required field: model` 로 거절된다. 성공해도 `session.updated` 같은 확인
   이벤트는 **오지 않는다**(무응답이 정상). 전사는 `transcription.delta` 로 오고 종료 이벤트가
   없어 정적 타임아웃으로 끊어야 한다.

### R3 — Qwen3 기준선의 "빈 전사" 는 남의 GPU 점유 탓이 아니었다

이전 두 런에서 기준선이 CER 1.0(빈 전사)으로 나와 "공용 GPU 경합" 으로 기록했으나 **오진이었다.**
`20260909_204118/gpu_watch.log` 가 반증한다: 21:38 에 카드가 완전히 비었고(free 23,686MiB) 그 뒤
올라온 두 PID 는 **AST 서버 자기 스택**이었다. 원인은 둘이고 둘 다 서버 구성 문제다.

**(1) ASR 엔진과 번역 모델이 한 카드에 올라간다.** AST 서버는 vLLM ASR(`gpu_memory_utilization`
기본 **0.8** = 18.6GiB)과 로컬 번역기 `google/madlad400-3b-mt`(fp16 **7.2GiB**, 첫 번역 호출 때
lazy 로딩)를 같은 24GB 카드에 올린다. 합이 넘어 `free=75MiB` → mm encoder 가 20MiB 를 할당하다
`torch.OutOfMemoryError`(`qwen3_asr.py:431`) → **EngineCore 사망** → 전 클립이
`ConnectionClosedOK: received 1000 (OK)` 로 끊긴다. **정상 종료 코드처럼 보이는 게 함정이다.**
드라이버가 `--gpu-memory-utilization` 을 넘기지 않고 있었다.

**(2) 번역이 `final` 전송을 막는다 — OOM 을 고쳐도 빈 전사가 남는다.** 파이프라인은
`[VAD-FINISH]` → 번역 → `[TRANS-VAD]` → `[AST-SEND] final` 순서다. madlad lazy 로딩·재시도가
클라이언트의 30초 idle 타임아웃을 넘기면 `final` 이 영영 안 나간다. 증상은 **서버 로그엔 완벽한
전사가 찍히는데 `AST-SEND` 카운트가 0**, 클라이언트 HYP 는 빈 문자열, CER 1.0.

**진단 지표**: `grep -c AST-SEND server.log` 가 0 이면 (2), 로그에 `EngineDeadError` 가 있으면 (1).
`KeyError: A`(`_slot()` / `stream_slots`)는 독립 버그가 아니라 (1)의 2차 증상이었다 — 엔진이 죽은
뒤 슬롯 미초기화 상태로 `finish` 가 불린 것으로, (1)을 고치자 0회가 됐다.

두 인자(`--gpu-memory-utilization 0.60`, `--trans-backend gtx --trans-retries 1`)를 드라이버에
박아 넣었다. `original`(ASR 전사)은 번역 실패에 영향받지 않으므로 **ASR 평가에는 이 조합이 정답**이다.

**결과 반전**: Qwen3 가 셋 중 가장 정확하다. 이전 런의 "Qwen3 참패" 는 전부 인프라 아티팩트였다.

### 환경 함정 (재발 방지)

- **`conda` 는 conda.sh 가 정의하는 셸 함수다.** `timeout conda run ...` / `env ... conda run ...`
  처럼 외부 명령 뒤에 두면 `failed to run command 'conda'` 로 죽는다 — 이걸로 런 하나를 날렸다.
  드라이버·스텝 스크립트는 전부 `$HOME/miniforge3/bin/conda` 를 쓴다.
- 비대화형 ssh 에는 conda 가 PATH 에 없다(같은 이유).
- 4090 의 tmux 는 `/usr/bin/tmux` 를 쓸 것. conda(stity) 쪽은 `server version is too old`.

---

## 10. 결론

1. **정확도는 Qwen3 압승.** 두 언어 모두 1위이고, en 에서는 Voxtral 의 4.6배, Nemotron 의 2.9배
   정확하다. 기준선을 교체할 이유가 없다.
2. **지연은 Nemotron 압도.** FSL 0.037~0.043s 로 Qwen3 의 1/10 수준이며 0.6B 로 ko CER 9.3% 를
   낸다. 지연이 정확도보다 중요한 경로(실시간 자막 프리뷰 등)의 후보로 남길 값어치가 있다.
3. **Voxtral 은 이 배포 형태로는 스트리밍 비교 대상이 아니다.** commit 트리거 동작이라 지연
   축에서 의미 있는 수치가 나오지 않는다. 정확도만 보면 ko 는 준수(0.0556)하나 en 이 약하다(0.1626).
4. **동작점은 곡선으로 골라야 한다.** rc0 사례가 그 근거다 — 기본값 한 점만 봤으면 "80ms 는 지연을
   벌어준다" 로 잘못 기록됐을 것이다.

---

## 11. 남은 일

- **전 데이터셋 풀 평가 미실시.** 지금 수치는 전부 서브셋(20~50클립) 스모크다. 4절 절차대로라면
  공통 동작점에서 한 번 풀런을 돌려야 결론이 확정된다.
- **클립 수 불균형.** Qwen3·Voxtral 은 20클립, Nemotron 스윕은 50클립이다. 표를 인용할 때 표본
  크기를 같이 적을 것. en rc13(20클립)과 rc6(50클립)은 직접 비교 불가다.
- **AST 종단 지연은 별도 측정이 필요하다.** 평가를 위해 번역 경로를 껐으므로 여기 FSL 은 ASR
  구간만의 값이다.
- **브랜치 정리.** `feat/multi-asr-backends` 는 `origin/main` 대비 4 커밋 뒤처져 있다. PR 전에
  리베이스가 필요하다.
- 산출물(런 디렉터리)은 워크트리 밖 `~/bench-results/` 에 있고 git 에 추적되지 않는다. 공용
  체크아웃에서 `git clean` 으로 산출물이 날아간 전력 때문에 벤치는 별도 워크트리에서 돌린다.

---

## 참고

- [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- [Voxtral Realtime (arXiv 2602.11298)](https://arxiv.org/abs/2602.11298)
- [mistralai/Voxtral-Mini-4B-Realtime-2602 · vLLM Recipes](https://recipes.vllm.ai/mistralai/Voxtral-Mini-4B-Realtime-2602)
- [OpenMDW-1.1 라이선스 원문](https://github.com/OpenMDW/OpenMDW/blob/main/1.1/LICENSE.OpenMDW-1.1)
- [kyutai/stt-1b-en_fr](https://huggingface.co/kyutai/stt-1b-en_fr)
- [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [Moonshine — specs, licence, downloads](https://speechtotext.dev/model/moonshine/)
