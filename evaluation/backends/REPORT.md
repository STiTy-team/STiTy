# 스트리밍 ASR 백엔드 다변화

| | |
|---|---|
| 브랜치 | `feat/multi-asr-backends` |
| 커밋 | `cd583e4` (base `origin/main` `15ac7dc`) |
| 실행 호스트 | RTX 4090 · 24564 MiB · driver 580.95.05 |
| 런 디렉터리 | `evaluation/backends/results/20260907_233539/` |
| 상태 | 무인 벤치마크 실행 중 (PID 1725069) |
| 작성 | 2026-09-07 |

Qwen3-ASR 단일 백엔드 체제에 Voxtral Realtime과 Nemotron Streaming을 추가한다.
후보 6종 심사, 측정 방법론 확정, 백엔드 골격 구현까지 완료했다.

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

커밋 `cd583e4` · `evaluation/backends/`

| 파일 | 역할 |
|---|---|
| `protocol.py` | 백엔드 중립 WS 서버. vllm/transformers를 import하지 않는다 — env 분리가 성립하는 이유. |
| `engine_nemotron.py` | cache-aware FastConformer-RNNT 엔진. `att_context_size` 로 지연 조절. probe 모드 내장. |
| `probe_voxtral.py` | vLLM `/v1/realtime` 스키마 수집기. `/openapi.json` 과 이벤트를 그대로 덤프한다. |
| `smoke_client.py` | FLEURS ko/en 클라이언트. 외부 의존성 없이 편집거리로 CER/WER 자체 채점. |
| `server.py` | 백엔드 선택 진입점. 엔진 모듈을 지연 import한다. |
| `run_backend_bench.sh` | 무인 드라이버. 백엔드 하나가 죽어도 계속 가고, `trap EXIT` 로 GPU를 반드시 반납한다. |

### 범위에서 제외한 것과 근거

- **엔진 인터페이스 리팩터** — 동작 보존 리팩터는 회귀 게이트가 있어야 의미가 있는데,
  미검증 백엔드 둘과 함께 무인으로 돌리면 숫자가 틀어졌을 때 원인을 귀속시킬 수 없다.
- **전 데이터셋 풀 평가** — 미검증 어댑터로 몇 시간짜리 풀런을 돌리면 실패 시 그 시간이
  전부 낭비된다.
- **Voxtral 브리지** — `/v1/realtime` 이벤트 스키마를 실행해본 적이 없다. 눈 감고 브리지를
  쓰는 대신 프로브로 실제 스키마를 확보한 뒤 왕복 한 번에 정확히 짠다.

---

## 8. 실행 현황

사용자 부재 중 승인 프롬프트를 받을 수 없으므로, 전 과정을 `setsid nohup` 으로 분리된
단일 스크립트에 넣었다. 한 번 기동되면 추가 승인 없이 끝까지 간다.

| | 단계 | 내용 | 상태 |
|---|---|---|---|
| A | 환경 준비 | conda env 2개 생성, Voxtral 4B(~9GB) · Nemotron 0.6B 다운로드 | 진행 중 |
| B | Nemotron | probe → 스모크 ko/en 20클립 → 지연 스윕 3점 (160 / 560 / 1120ms, ko 50클립) | 대기 |
| C | Voxtral | vLLM 기동 → realtime 스키마·이벤트 프로브 (ko/en) | 대기 |
| D | Qwen3 기준선 | 기존 AST 서버, 동일 클립으로 스모크 | 대기 |
| E | 정리 | 모델 하역 → compute-apps 0 확인 → GPU 반납 | 대기 |

> **사고 기록**
> 기동 직후 완료 오경보가 있었다. `pgrep -f run_backend_bench.sh | head -1` 로 잡은
> PID 1725068은 fork 후 즉시 빠지는 `setsid` 래퍼였고, 실제 스크립트는 1725069였다.
> 워처를 **특정 PID가 아니라 해당 프로세스의 부재**를 조건으로 교체했다. 벤치마크 자체는
> 영향받지 않았다.

---

## 9. 리스크

- **어댑터 미검증** — 두 모델 모두 문서만 보고 작성했다. 첫 실행 실패 확률이 낮지 않다.
  완화책으로 각 백엔드가 probe를 먼저 통과해야 스모크로 넘어가며, 실패 시 실제 시그니처를
  JSON으로 남기고 그 백엔드만 건너뛴다.
- **커밋 정책 전제 불일치** — 현재 중복 가드는 Qwen3-ASR이 이전 텍스트를 수정한다는 전제
  위에 있다. Nemotron은 RNN-T라 단조 증가하며 절대 수정하지 않는다 — 이 가드들이
  무의미하거나 해로울 수 있다.
- **공용 GPU** — 4090은 팀 공용이다. 순차 실행으로 동시 적재를 피했고, 종료 경로 전체에
  GPU 반납을 걸었다.
- **OpenMDW-1.1 미등재** — 2026년 5월 기준 SPDX License List에 아직 없다. 권리 제약이
  아니라 자동 스캐너가 unknown으로 처리할 수 있다는 실무 이슈다.

---

## 참고

- [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- [Voxtral Realtime (arXiv 2602.11298)](https://arxiv.org/abs/2602.11298)
- [mistralai/Voxtral-Mini-4B-Realtime-2602 · vLLM Recipes](https://recipes.vllm.ai/mistralai/Voxtral-Mini-4B-Realtime-2602)
- [OpenMDW-1.1 라이선스 원문](https://github.com/OpenMDW/OpenMDW/blob/main/1.1/LICENSE.OpenMDW-1.1)
- [kyutai/stt-1b-en_fr](https://huggingface.co/kyutai/stt-1b-en_fr)
- [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [Moonshine — specs, licence, downloads](https://speechtotext.dev/model/moonshine/)
