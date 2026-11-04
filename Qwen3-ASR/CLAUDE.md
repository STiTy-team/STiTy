# Qwen3-ASR (vendored upstream)

upstream(QwenLM/Qwen3-ASR) 코드를 STiTy 리포가 **직접 추적**한다. git submodule이 아니라서 이 안의 파일 변경은 다른 파일과 똑같이 부모 리포 커밋에 포함된다 (`git add Qwen3-ASR/...`). 일반 사용법은 README.md를 보고, 이 파일은 STiTy 통합 사항만 다룬다.

## STiTy 통합 진입점

- **추론 엔진**: `qwen_asr/` — `core/components/transcription/` 의 `qwen3`·`qwen-seg` 부품이 이것을 불러 쓴다
- **파인튜닝**: `finetuning/qwen3_asr_sft.py` (SFT) + `finetuning/utils/` (병합). 병합된 가중치는 여기 두지 않고 리포 루트 `models/`에 있다 — `models/Qwen3-ASR-1.7B-{en,ko}-silence-*-merged` 와 `models/Qwen3-ASR-1.7B-en-dailytalk-seg`.

## upstream 동기화 규칙

- 여기 있는 파일에는 STiTy 쪽 수정이 섞여 있다(예: `qwen_asr/inference/`, `finetuning/qwen3_asr_sft.py`). upstream을 통째로 덮어쓰면 날아간다.
- 자동 동기화 명령은 없다. 필요한 파일만 골라 가져와 수동 병합할 것.

## 설치

```bash
pip install -e ./Qwen3-ASR              # transformers 백엔드
pip install -e "./Qwen3-ASR[vllm]"     # vLLM 포함
```

## 패키지 구조

| 디렉토리 | 역할 |
|---|---|
| `qwen_asr/core/` | 모델 백엔드 (transformers/vllm) |
| `qwen_asr/inference/` | ASR 추론 엔진 |
| `qwen_asr/cli/` | CLI 진입점 (qwen-asr-serve 등) |
| `examples/` | upstream 사용 예제 |
| `finetuning/` | SFT 스크립트 및 병합 유틸리티 |

## 테스트

`tests/test_dot_commit_boundary.py` — 마지막 문장의 마침표를 경계로 잡되 소수점·약어는 제외하는
판정(`41268ba`)을 붙잡아 둔 회귀 테스트다. 모델도 GPU도 필요 없다.

```bash
PYTHONPATH=$PWD/Qwen3-ASR python Qwen3-ASR/tests/test_dot_commit_boundary.py
```
