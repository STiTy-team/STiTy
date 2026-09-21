# STiTy Bench의 다중 ASR 런타임 의존성 관리 설계

> 대상 환경: Linux, NVIDIA GeForce RTX 4090 24GB 1대  
> 목적: 의존성이 서로 다른 다양한 ASR 모델을 동일한 config 기반 bench에서 반복적이고 재현 가능하게 평가한다.

## 1. 요약

현재 STiTy bench는 하나의 `config.yml`에 데이터셋, 언어 방향, ASR, 번역기, VAD 및 commit 정책을 선언하면 registry가 해당 컴포넌트를 찾아 한 Python 프로세스 안에서 pipeline을 조립한다. 이 구조는 실험 정의가 명시적이고 결과 재현이 쉽다는 장점이 있지만, ASR 모델마다 요구하는 `torch`, `transformers`, `vllm`, NeMo 및 CUDA 관련 버전이 달라지면 하나의 Python 환경으로는 더 이상 모든 모델을 안정적으로 실행할 수 없다.

실제 후보 세 개만 보더라도 충돌이 이미 발생한다.

| ASR backend | 주요 runtime 요구사항 | streaming 경로 |
|---|---|---|
| Qwen3-ASR 1.7B 현재 기준선 | `transformers==4.57.6`, `vllm==0.14.0` | STiTy가 사용 중인 Qwen vLLM SDK |
| Voxtral Mini 4B Realtime | `vllm>=0.20`, `mistral-common[audio]>=1.9` | vLLM Realtime WebSocket API |
| Nemotron 3.5 ASR Streaming 0.6B | `transformers>=5.13` 또는 NeMo 26.06 | Transformers/NeMo native streaming |

따라서 단일 환경에 모든 optional dependency를 설치하는 방식은 제외한다. 권장 방향은 다음과 같다.

1. 환경을 **모델 checkpoint별이 아니라 runtime compatibility family별**로 나눈다.
2. config는 지금처럼 논리적인 backend와 모델을 선택하고, 실제 환경은 중앙 manifest가 결정한다.
3. 얇은 launcher가 config를 읽고 올바른 환경의 Python으로 bench를 재실행한다.
4. ASR와 충돌할 수 있는 로컬 번역기는 별도 환경과 프로세스로 분리한다.
5. lockfile과 실제 실행 환경 fingerprint를 결과에 남긴다.

이 설계에는 uv와 Conda를 모두 사용할 수 있다. 이 프로젝트에서는 **독립 uv project + 환경별 `uv.lock`**을 우선 권장한다. 모든 실행이 Linux 4090 한 대에서 이루어지고 현재 의존성 대부분이 PyPI 중심이기 때문에 설정과 반복 실행이 더 단순하고 빠르다. Conda는 CUDA 및 비 Python 라이브러리까지 환경별로 강하게 관리해야 하거나 팀의 기존 운영 표준이 Conda일 때 적합하다.

---

## 2. 배경

STiTy는 실시간 다국어 음성 번역 시스템이다. bench는 실제 앱 개발 과정에서 ASR 모델, 번역 모델, 데이터셋, 언어 방향 및 commit 정책을 반복해서 바꿔 평가하기 위해 만들어졌다.

현재 bench의 핵심 흐름은 다음과 같다.

```text
config.yml
    ↓
Pydantic 기반 config 검증
    ↓
registry에서 pipeline과 component 선택
    ↓
ASR + VAD + translation을 같은 Python 프로세스에서 조립
    ↓
실시간 속도로 audio 공급
    ↓
events.jsonl / items.jsonl / summary.json 생성
```

이 설계가 지키는 중요한 원칙은 다음과 같다.

- 실험 조건은 config에 명시한다.
- 알 수 없는 옵션은 조용히 무시하지 않고 실행 전에 오류로 처리한다.
- config 검증은 GPU 모델 load보다 먼저 수행한다.
- config 원본과 실제 resolved 값을 결과에 남긴다.
- dataset, metric 및 report 코드는 특정 ASR 구현을 알지 않는다.
- ASR 구현 차이는 `Transcriber` adapter가 흡수한다.

현재 Qwen adapter도 무거운 `qwen_asr`, `vllm` import를 `load()`까지 미룬다. 이는 모델 의존성과 나머지 bench 코드를 분리하려는 기존 철학에 부합한다.

## 3. 문제의식

### 3.1 단일 Python 환경은 서로 다른 ASR runtime을 격리하지 못한다

Python dependency resolver는 한 환경 안에서 패키지별로 하나의 버전만 선택한다. 예를 들어 Qwen 기준선이 요구하는 `transformers==4.57.6`과 Nemotron이 요구하는 `transformers>=5.13`을 동시에 만족시킬 수 없다. Qwen의 `vllm==0.14.0`과 Voxtral의 `vllm>=0.20`도 같은 문제를 가진다.

Optional dependency나 dependency group은 설치할 패키지를 선택하는 기능이지, 이미 실행 중인 Python 프로세스 안에서 패키지 버전을 격리하는 기능이 아니다.

### 3.2 config를 읽은 뒤 현재 Python 환경을 교체할 수 없다

가상환경은 Python 프로세스가 시작될 때 결정된다. Qwen 환경에서 시작한 bench가 config를 읽은 뒤 같은 프로세스 안에서 Nemotron 환경으로 전환할 수는 없다.

따라서 환경 선택은 반드시 실제 bench 프로세스가 시작되기 전에 이루어져야 한다.

```text
config를 읽는 launcher
    ↓ backend에 필요한 runtime 결정
선택된 환경의 Python으로 새 프로세스 실행
    ↓
기존 bench 실행
```

### 3.3 모델 수와 환경 수를 일대일로 늘리면 관리 비용이 커진다

Checkpoint가 다르더라도 같은 SDK와 dependency 범위를 사용한다면 환경을 공유할 수 있다.

예를 들어 다음 모델들은 일반적으로 하나의 adapter/runtime family로 묶을 수 있다.

- Qwen3-ASR 0.6B, 1.7B 및 같은 코드 기반의 fine-tuned checkpoint
- Whisper tiny부터 large까지의 faster-whisper 호환 checkpoint
- 동일한 Transformers `AutoModelForRNNT` streaming API를 사용하는 모델
- 동일한 NeMo cache-aware streaming API를 사용하는 모델

따라서 관리 단위는 다음 세 계층으로 나누어야 한다.

```text
model checkpoint
    ↓ 사용
backend adapter
    ↓ 실행됨
runtime environment
```

### 3.4 번역기 의존성이 ASR 비교를 오염시킬 수 있다

현재 로컬 번역기도 `torch`와 `transformers`를 사용한다. ASR에 맞추기 위해 Transformers 버전을 바꾸면 번역기의 출력, 메모리 사용량 또는 latency까지 달라질 수 있다. 그러면 ASR 비교 실험에서 발생한 차이와 번역 runtime 변화가 하나의 결과에 섞인다.

다양한 ASR를 공정하게 비교하려면 다음 두 단계를 분리하는 것이 좋다.

1. 번역을 끈 ASR-only 평가
2. 별도 고정 환경의 translation server를 연결한 end-to-end 평가

코드베이스에는 이미 HTTP 기반 `RemoteTranslator`가 있으므로 이를 bench registry에 연결하면 된다.

### 3.5 모델마다 공식 streaming interface가 다르다

세 후보 모델의 streaming 형태도 동일하지 않다.

- Qwen은 SDK의 streaming state와 SEG/dot callback을 사용한다.
- Voxtral은 공식적으로 vLLM `/v1/realtime` WebSocket 경로를 권장한다.
- Nemotron Transformers 구현은 audio feature generator와 `TextIteratorStreamer`를 사용한다.

의존성 격리만으로는 부족하며, 각 구현의 출력을 공통 `partial`과 `transcribed` 이벤트로 정규화하는 adapter가 필요하다.

---

## 4. 공통 설계 원칙

uv와 Conda 중 무엇을 선택해도 다음 구조는 동일하게 유지한다.

### 4.1 환경 단위는 runtime compatibility family다

초기 runtime family 예시는 다음과 같다.

```text
qwen3-vllm014
vllm020-realtime
transformers513-rnnt
nemo2606
faster-whisper
onnxruntime-gpu
translation
```

새 모델을 추가할 때는 다음 순서로 판단한다.

| 조건 | 필요한 작업 |
|---|---|
| 기존 adapter와 기존 runtime에서 동작 | config만 추가 |
| 기존 adapter를 쓸 수 있지만 dependency가 충돌 | runtime 환경만 추가 |
| 새로운 streaming state/API 사용 | adapter와 runtime 추가 |
| 외부 API 또는 공식 model server 사용 | client adapter 추가, 필요하면 managed server lifecycle 추가 |

### 4.2 config에는 환경 경로나 환경 이름을 직접 적지 않는다

사용자가 다음과 같이 서로 맞지 않는 조합을 만들 수 있게 하면 안 된다.

```yaml
# 권장하지 않는 형태
transcription:
  name: qwen3-vllm
  environment: transformers513-rnnt
```

config에는 실험 대상만 쓴다.

```yaml
transcription:
  name: qwen3-vllm
  model: Qwen/Qwen3-ASR-1.7B
```

backend와 runtime의 연결은 repository가 관리하는 manifest에 둔다.

```toml
[transcription.qwen3-vllm]
runtime = "qwen3-vllm014"
module = "core.pipelines.transcription.qwen3"

[transcription.voxtral-realtime]
runtime = "vllm020-realtime"
module = "core.pipelines.transcription.voxtral"

[transcription.hf-rnnt-streaming]
runtime = "transformers513-rnnt"
module = "core.pipelines.transcription.hf_rnnt"
```

### 4.3 기준선 runtime을 제자리에서 업그레이드하지 않는다

Qwen 기준선의 vLLM을 0.14에서 0.20으로 올리면 모델 weight가 같더라도 streaming, token emission, latency 및 commit 결과가 달라질 수 있다. 기존 lockfile을 덮어쓰는 대신 별도 backend/runtime으로 취급한다.

```text
qwen3-vllm014  # 기존 기준선
qwen3-vllm020  # runtime 업그레이드 실험
```

### 4.4 결과에 실행 환경을 기록한다

`summary.json`에 다음 정보를 추가한다.

```json
{
  "runtime": {
    "id": "transformers513-rnnt",
    "lock_sha256": "...",
    "python": "3.12.x",
    "torch": "...",
    "transformers": "5.13.x",
    "vllm": null,
    "cuda_runtime": "...",
    "nvidia_driver": "...",
    "gpu": "NVIDIA GeForce RTX 4090",
    "model_id": "nvidia/nemotron-3.5-asr-streaming-0.6b",
    "model_revision": "...",
    "execution_mode": "in_process"
  }
}
```

Config만 보관해서는 dependency와 weight revision 변화를 재현할 수 없다. lockfile hash와 model revision까지 있어야 동일 결과인지 판단할 수 있다.

### 4.5 adapter module은 선택될 때만 import한다

현재처럼 transcription package의 `__init__.py`가 모든 adapter를 import하는 방식은 모델이 많아지면 위험하다. 선택하지 않은 adapter의 dependency가 없다는 이유로 config 검증 자체가 실패할 수 있기 때문이다.

Registry에 lazy entry를 등록하고 `registry.get(name)` 시점에 선택된 module만 import하도록 변경한다.

```python
transcribers.register_lazy(
    name="voxtral-realtime",
    module="core.pipelines.transcription.voxtral",
    class_name="VoxtralRealtimeTranscription",
)
```

### 4.6 모델별 출력을 공통 event contract로 변환한다

Adapter의 공통 계약은 다음 정도로 유지한다.

```text
load()
start(language)
transcribe(pcm_chunk)
    → partial: 현재 전체 미확정 문자열
    → transcribed: 새로 확정된 문자열
flush(reason)
finish()
close()
```

모델 고유의 delta, token streamer, state text 및 final event는 adapter 안에서 기존 bench event 형태로 바꾼다. `recv_elapsed_sec`은 bench가 결과를 실제로 받은 시점을 기준으로 기록하여 in-process와 localhost model server 경로를 모두 사용자 관점에서 비교할 수 있게 한다.

---

## 5. uv를 사용한 해결 설계

### 5.1 구조

각 runtime family를 독립된 uv project로 둔다.

```text
STiTy/
├─ pyproject.toml
├─ bench/
│  ├─ launch.py
│  └─ runtimes.toml
├─ runtimes/
│  ├─ qwen3-vllm014/
│  │  ├─ pyproject.toml
│  │  └─ uv.lock
│  ├─ vllm020-realtime/
│  │  ├─ pyproject.toml
│  │  └─ uv.lock
│  ├─ transformers513-rnnt/
│  │  ├─ pyproject.toml
│  │  └─ uv.lock
│  ├─ faster-whisper/
│  └─ translation/
└─ core/pipelines/transcription/
```

서로 충돌하는 환경을 하나의 uv workspace로 묶지 않는다. 각 디렉터리가 독립적인 `pyproject.toml`, `.venv`, `uv.lock`을 가진다.

### 5.2 루트 package

루트 `pyproject.toml`에는 bench와 core가 공통으로 사용하는 가벼운 dependency만 선언한다.

```toml
[project]
name = "stity-bench"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = [
    "pyyaml>=6",
    "pydantic>=2",
    "numpy",
    "soundfile",
    "librosa",
    "jiwer",
    "sacrebleu[ja,ko]>=2.4",
    "rich>=13",
    "aiohttp",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["bench*", "core*"]
```

`torch`, `transformers`, `vllm`, NeMo 및 모델별 tokenizer package는 루트에서 제외한다.

### 5.3 runtime project 예시

Qwen 기준선 환경:

```toml
[project]
name = "stity-runtime-qwen3-vllm014"
version = "0.1.0"
requires-python = "==3.12.*"
dependencies = [
    "stity-bench",
    "qwen-asr[vllm]",
]

[tool.uv.sources]
stity-bench = { path = "../..", editable = true }
qwen-asr = { path = "../../Qwen3-ASR", editable = true }

[tool.uv]
package = false
```

Nemotron Transformers 환경:

```toml
[project]
name = "stity-runtime-transformers513-rnnt"
version = "0.1.0"
requires-python = "==3.12.*"
dependencies = [
    "stity-bench",
    "transformers>=5.13,<5.14",
    "accelerate",
    "torch==<검증된 버전>",
]

[tool.uv.sources]
stity-bench = { path = "../..", editable = true }

[tool.uv]
package = false
```

PyTorch 및 CUDA wheel source도 검증된 값으로 고정한다. 벤치마크 재현성이 목적이므로 실행할 때마다 driver를 보고 다른 wheel을 선택하는 자동 모드는 피한다.

### 5.4 launcher

`bench/launch.py`는 raw YAML에서 transcription backend 이름만 읽고 manifest에서 runtime을 찾은 뒤 해당 환경으로 재실행한다.

```text
make bench CONFIG=config.yml
    ↓
가벼운 launcher가 config.yml 읽기
    ↓
transcription.name → runtime lookup
    ↓
uv run --project runtimes/<runtime> --locked python -m bench config.yml
```

사용자 명령은 기존과 동일하게 유지한다.

```bash
make bench CONFIG=configs/nemotron-ko-en.yml
```

`--locked`를 기본값으로 사용한다. 개발자가 runtime 환경에 `pip install`로 패키지를 임시 추가하는 경로는 허용하지 않는다. dependency 변경은 `pyproject.toml` 수정과 `uv lock` 갱신으로만 수행한다.

### 5.5 환경 관리 명령

```bash
# runtime 최초 생성 또는 dependency 변경 후
cd runtimes/transformers513-rnnt
uv lock
uv sync --locked

# 해당 runtime에서 직접 smoke test
uv run --project runtimes/transformers513-rnnt --locked \
  python -m bench configs/nemotron-smoke.yml

# CI에서 lockfile 최신 상태 확인
uv lock --project runtimes/transformers513-rnnt --check
```

### 5.6 uv 방식의 장점

- PyPI 기반 ML runtime 설치와 반복 동기화가 빠르다.
- 환경마다 정확한 lockfile을 repository에 보관할 수 있다.
- 별도의 shell activation 없이 `uv run --project ...`로 실행할 수 있다.
- 로컬 STiTy와 vendored Qwen package를 editable path dependency로 연결하기 쉽다.
- 환경 이름이 사용자의 전역 Conda 상태가 아니라 repository 디렉터리에 귀속된다.
- launcher와 결합하면 사용자는 어떤 환경을 활성화해야 하는지 알 필요가 없다.
- 한 runtime이 깨져도 다른 runtime의 `.venv`에는 영향이 없다.

### 5.7 uv 방식의 단점과 트레이드오프

- CUDA driver, `ffmpeg`, `libsndfile` 같은 system dependency는 기본적으로 host가 관리해야 한다.
- runtime마다 독립 lockfile이 생기므로 공통 dependency 정보가 일부 반복된다.
- 수많은 runtime lockfile을 한 번에 업그레이드하면 검증 비용이 크다.
- FlashAttention, custom CUDA extension처럼 build isolation에 민감한 package는 별도 설정이 필요할 수 있다.
- 서로 다른 runtime project 사이의 공통 package 버전 일치를 uv가 자동 보장하지 않는다. 공통 dependency 정책은 CI나 별도 검사로 확인해야 한다.

Linux 4090 한 대라는 현재 조건에서는 system dependency 종류가 제한적이고 host를 통제할 수 있으므로 이러한 단점은 비교적 작다.

---

## 6. Conda를 사용한 해결 설계

### 6.1 구조

Conda 방식도 runtime family별 environment를 만든다는 원칙은 동일하다.

```text
STiTy/
├─ environment/
│  ├─ qwen3-vllm014.yml
│  ├─ vllm020-realtime.yml
│  ├─ transformers513-rnnt.yml
│  ├─ faster-whisper.yml
│  └─ translation.yml
├─ locks/
│  ├─ qwen3-vllm014-linux-64.lock.yml
│  ├─ vllm020-realtime-linux-64.lock.yml
│  └─ transformers513-rnnt-linux-64.lock.yml
├─ bench/
│  ├─ launch.py
│  └─ runtimes.toml
└─ core/
```

Manifest는 backend를 Conda 환경 이름이나 repository-local prefix에 연결한다.

```toml
[transcription.qwen3-vllm]
runtime = "qwen3-vllm014"
conda_prefix = ".conda/qwen3-vllm014"

[transcription.voxtral-realtime]
runtime = "vllm020-realtime"
conda_prefix = ".conda/vllm020-realtime"
```

전역 이름 기반 환경보다 repository-local prefix를 권장한다. 다른 checkout 또는 다른 프로젝트의 동일한 환경 이름과 충돌하는 것을 피할 수 있기 때문이다.

### 6.2 environment 정의 예시

```yaml
name: stity-nemotron-transformers513
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.12
  - ffmpeg
  - libsndfile
  - pip
  - pip:
      - -e ../..
      - transformers>=5.13,<5.14
      - accelerate
      - torch==<검증된 버전>
```

다만 `environment.yml`의 범위 선언만으로는 시간이 지난 뒤 정확히 같은 환경이 재생성된다고 보장하기 어렵다. 실제 사용에는 다음 중 하나가 추가되어야 한다.

- 플랫폼별 explicit spec
- conda-lock 같은 별도 lock 생성 도구
- 검증된 package build까지 고정한 환경 export

### 6.3 launcher

Conda 환경도 사용자가 직접 activate하지 않는다.

```bash
conda run \
  --prefix .conda/transformers513-rnnt \
  --no-capture-output \
  python -m bench configs/nemotron-ko-en.yml
```

Launcher가 config와 manifest를 읽고 위 명령을 실행한다. `conda activate`를 실행 절차에 포함하면 사용자의 현재 shell 상태에 따라 잘못된 환경에서 bench가 실행될 수 있다.

환경 생성과 갱신 속도가 중요하다면 Conda 대신 Mamba 또는 Micromamba를 같은 manifest/lock 구조의 실행기로 사용할 수 있다.

### 6.4 Conda 방식의 장점

- Python 외의 `ffmpeg`, `libsndfile`, C/C++ library 및 CUDA toolkit package를 같은 환경 정의에 포함하기 쉽다.
- ML 연구 환경에서 익숙한 사용자가 많다.
- NeMo와 같이 Python package 외 시스템 요구사항이 많은 stack에 유리하다.
- channel과 package build까지 고정하면 binary dependency를 강하게 통제할 수 있다.
- NVIDIA 또는 연구 모델의 공식 설치 문서가 Conda 환경을 전제로 하는 경우 적용이 쉽다.

### 6.5 Conda 방식의 단점과 트레이드오프

- 환경 solve와 생성이 uv보다 느린 편이다.
- `environment.yml`만으로는 완전한 lockfile 역할을 하지 못한다.
- Conda와 `pip:` dependency를 혼합하면 어느 resolver가 최종 상태를 책임지는지 불명확해질 수 있다.
- 전역 environment name과 수동 activation은 repository와 환경 사이의 연결을 약하게 만든다.
- platform-specific explicit lock은 강한 재현성을 주지만 Linux/CUDA 환경에 종속된다.
- 팀원이 임의로 `conda install` 또는 `pip install`하면 선언 파일과 실제 환경이 쉽게 달라진다.
- 다수의 작은 runtime을 자주 만들고 버리는 workflow에는 상대적으로 무겁다.

---

## 7. uv와 Conda 비교

| 평가 항목 | uv 독립 project | Conda/Mamba 독립 environment |
|---|---|---|
| 현재 config/registry 철학과의 적합성 | 높음 | 높음 |
| 반복 실행 편의성 | 매우 높음 | 보통~높음 |
| 환경 생성 및 sync 속도 | 빠름 | 상대적으로 느림, Mamba로 개선 가능 |
| Python dependency lock | `uv.lock`으로 직접 제공 | 추가 lock 전략이 필요 |
| 비 Python dependency 관리 | host 또는 container에 의존 | 강함 |
| PyPI 중심 최신 모델 대응 | 강함 | pip 혼합이 자주 필요 |
| CUDA extension 대응 | 가능하나 별도 build 설정 필요 | Conda package가 있으면 편리함 |
| repository 귀속성 | project-local `.venv`로 명확함 | prefix 환경을 쓰면 명확함 |
| 전역 환경 오염 위험 | 낮음 | 이름 기반 환경 사용 시 존재 |
| 신규 runtime 추가 비용 | 낮음 | 중간 |
| Linux 4090 한 대에서의 적합성 | 가장 높음 | 충분히 가능하지만 다소 무거움 |

### 핵심 트레이드오프

uv는 **Python runtime을 빠르고 명시적으로 격리하는 데 최적화**되어 있다. 반면 Conda는 **Python을 포함한 더 넓은 binary/system stack을 함께 관리하는 데 강점**이 있다.

현재 후보 모델들의 공식 실행 경로는 대부분 pip/uv로 설치할 수 있고, 실행 머신이 고정된 Linux 4090 한 대이므로 host의 NVIDIA driver와 소수 system package를 표준화하기 어렵지 않다. 따라서 uv의 단순성과 속도가 더 큰 이점이다.

반대로 다음 조건이 생기면 Conda/Mamba의 가치가 커진다.

- 모델마다 서로 다른 CUDA toolkit이나 컴파일러를 요구한다.
- NeMo 기반 학습·fine-tuning 환경이 주요 workflow가 된다.
- 연구실의 다른 프로젝트와 GPU 환경이 이미 Conda로 표준화되어 있다.
- PyPI wheel이 없는 native dependency를 자주 빌드해야 한다.

## 8. RTX 4090 한 대에서의 운영 원칙

환경을 분리해도 GPU 메모리는 물리적으로 공유된다. 다음 규칙이 필요하다.

1. 벤치마크 중 ASR runtime은 하나만 실행한다.
2. 이전 ASR subprocess가 완전히 종료되고 GPU 메모리가 반환된 뒤 다음 실행을 시작한다.
3. ASR-only 평가에서는 translation server를 실행하지 않는다.
4. End-to-end 평가에서는 ASR을 먼저 load하고 translation server를 나중에 시작한다.
5. Voxtral은 기본 131K context 대신 실제 최대 세션에 맞는 `max_model_len`을 사용한다.
6. vLLM의 `gpu_memory_utilization`과 실제 peak VRAM을 config 및 summary에 함께 기록한다.
7. Model load/warmup 시간과 streaming inference latency를 분리해서 측정한다.

특히 Voxtral은 BF16 기준 최소 16GB GPU가 권장되므로 24GB 4090에서 번역기와 함께 실행할 때 메모리 여유가 크지 않다. 반면 0.6B Nemotron은 번역기와의 GPU 공유 및 높은 동시성 측면에서 유리할 가능성이 높다. 이는 parameter 수로부터의 예상일 뿐이며 실제 peak VRAM은 동일한 bench에서 측정해야 한다.

## 9. 공정한 모델 비교를 위한 추가 변경

### 9.1 공통 VAD commit 모드

Qwen에는 native SEG가 있지만 다른 모델에는 동일한 기능이 없다. 순수 ASR 비교와 실제 최적 UX 비교를 분리하기 위해 다음 두 축을 둔다.

- `commit: vad`: 모든 native segment commit을 끄고 동일한 VAD에서만 확정
- 모델 권장 모드: Qwen SEG, Voxtral realtime delta, Nemotron streaming + VAD 등 각 모델의 최적 경로 사용

현재 commit 설정에 `vad` 모드를 추가하는 것이 좋다.

```python
"vad": {
    "always_commit": False,
    "enable_dot_commit": False,
    "hide_seg": True,
}
```

### 9.2 언어 조건 비교

언어를 알려주는 방식도 모델마다 다르다.

- Qwen: allowed language restriction
- Nemotron: 특정 locale conditioning 또는 auto detection
- Voxtral: Realtime API가 제공하는 범위 안에서 자동 처리

가능한 모델은 다음 두 조건을 분리해서 측정한다.

- `auto`: 모델이 언어를 감지
- `oracle`: dataset의 실제 source language를 모델에 제공

언어 감지 성능은 routing metric으로 별도 기록하며, oracle 결과와 auto 결과를 같은 WER 열에 섞지 않는다.

### 9.3 추가할 측정값

기존 WER/CER/FSL/LAAL 외에 다음 값이 유용하다.

- 첫 non-empty partial latency
- partial revision 횟수
- 이미 노출된 문자열이 수정된 문자 비율
- real-time factor
- peak GPU memory
- model load 및 warmup 시간
- 빈 hypothesis 비율
- execution mode: in-process, managed server 또는 remote API

## 10. 도입 단계

### 1단계: 현재 Qwen 기준선 동결

- 현재 Qwen dependency를 uv project로 옮긴다.
- `uv.lock`을 커밋한다.
- 기존 config가 같은 결과를 내는지 확인한다.
- runtime fingerprint를 summary에 추가한다.

### 2단계: launcher와 lazy registry 도입

- `bench/runtimes.toml`을 추가한다.
- `bench/launch.py`를 추가한다.
- `make bench`가 launcher를 사용하도록 변경한다.
- 선택된 adapter만 import하도록 registry를 변경한다.

### 3단계: 번역 환경 분리

- `translation: remote` component를 추가한다.
- translation 전용 uv project와 lockfile을 만든다.
- ASR-only와 end-to-end 실행을 분리한다.

### 4단계: Nemotron 통합

- `transformers513-rnnt` runtime을 만든다.
- 먼저 공식 Transformers streaming 경로를 adapter로 구현한다.
- NeMo는 fine-tuning이나 NeMo 전용 최적화가 필요해질 때 별도 runtime으로 추가한다.

### 5단계: Voxtral 통합

- `vllm020-realtime` runtime을 만든다.
- Adapter가 `vllm serve` subprocess와 WebSocket lifecycle을 관리한다.
- 종료 및 예외 시 자식 프로세스와 GPU 메모리가 확실히 정리되는지 검증한다.

### 6단계: 신규 모델 추가 절차 표준화

각 신규 모델 PR에 다음 항목을 요구한다.

- 재사용할 adapter 및 runtime 또는 새로 추가한 이유
- 공식 dependency와 지원 streaming 경로
- lockfile 변경
- model revision
- 최소 smoke config
- ASR-only 결과
- peak VRAM과 real-time factor
- 지원 언어 및 language conditioning 방식
- license 및 사용 제한

## 11. 최종 권고

현재 조건에서는 다음 구성을 채택하는 것이 가장 합리적이다.

```text
환경 관리자       uv
환경 격리 단위    runtime compatibility family
환경 선택         config를 읽는 launcher가 자동 결정
공통 코드         root stity-bench package
ASR 구현          lazy-loaded Transcriber adapter
번역 모델         별도 uv 환경의 translation server
재현성            runtime별 uv.lock + summary fingerprint
GPU 운용           ASR 하나씩 순차 실행, translation은 별도 단계
```

Conda는 대체 설계로 유지할 수 있지만, 현재처럼 고정된 Linux 4090 머신에서 다양한 pip 기반 ASR를 빠르게 교체하는 목적에는 uv가 더 간결하다. 향후 NeMo 학습 환경이나 복잡한 native/CUDA build가 핵심이 되면 해당 runtime만 Conda/Mamba 또는 container로 실행하는 혼합 방식도 가능하다. 중요한 것은 모든 runtime을 하나의 환경 관리 도구로 억지로 통일하는 것이 아니라, config와 결과 형식은 통일하되 실행 환경은 격리하는 것이다.

## 12. 참고 자료

- [Qwen3-ASR 공식 모델 카드](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [Voxtral Mini 4B Realtime 공식 모델 카드](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
- [Voxtral vLLM 공식 recipe](https://recipes.vllm.ai/mistralai/Voxtral-Mini-4B-Realtime-2602)
- [Nemotron 3.5 ASR Streaming 공식 모델 카드](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- [uv locking과 syncing](https://docs.astral.sh/uv/concepts/projects/sync/)
- [uv workspace와 독립 project 선택 기준](https://docs.astral.sh/uv/concepts/projects/workspaces/)
- [Conda `run` 명령](https://docs.conda.io/projects/conda/en/latest/commands/run.html)

