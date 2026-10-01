#!/usr/bin/env bash
# WhisperLiveKit 서버 기동. 조합 하나 = 프로세스 하나. env 는 asr-wlk (기존 env 와 분리).
#
#   bash run_wlk_server.sh <simul|local|funasr> [port=8791] [wlk 추가 인자...]
#
# 조합 (모델 / 출력 정책):
#   simul   Whisper large-v3 + SimulStreaming(AlignAtt).  --backend faster-whisper 는 **인코더**만
#           CTranslate2 로 돌린다는 뜻이고 디코더는 openai whisper large-v3.pt 다
#           (simul_whisper/backend.py: fw_encoder=WhisperModel(...) + load_model(decoder_only=True)).
#   local   Whisper large-v3 + LocalAgreement. 순수 faster-whisper(CTranslate2) 경로.
#   funasr  FunASR SenseVoiceSmall + LocalAgreement. 한국어 네이티브 경량 모델.
#
# 공통: --pcm-input (s16le 16kHz 바이트를 그대로 받음, ffmpeg 우회), --lan auto (언어 힌트 없음),
#       127.0.0.1 바인드. 지연 노브(--min-chunk-size, --frame-threshold, --audio-max-len ...)는
#       전부 WLK 기본값 — "기본값 한 점" 비교다.
set -uo pipefail
COMBO="${1:?combo: simul|simul-nf|simul-c1|local|funasr}"; PORT="${2:-8791}"; shift $(( $# >= 2 ? 2 : 1 ))
ENV="$HOME/miniforge3/envs/asr-wlk"
WLK="$ENV/bin/wlk"
# CTranslate2(faster-whisper) 는 libcublas.so.12 / libcudnn.so.9 를 soname 으로 dlopen 한다.
# torch 가 먼저 import 되면 같은 라이브러리가 이미 프로세스에 올라와 있어 보통 찾지만,
# 경로를 명시해 두면 import 순서에 기대지 않는다. (env 는 torch cu128 로 맞춰 둔 상태다 —
# 기본 pip 의 cu130 torch 로는 libcublas.so.12 가 없어 웜업에서 죽었다, 2026-09-22.)
NV="$ENV/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$NV/cublas/lib:$NV/cudnn/lib:$NV/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
COMMON=(--host 127.0.0.1 --port "$PORT" --pcm-input --lan auto)
case "$COMBO" in
  simul)  ARGS=(--model large-v3 --backend faster-whisper --backend-policy simulstreaming) ;;
  # simul + --never-fire. large-v3 용 CIF(단어 끝 감지) 체크포인트가 없어 기본값은 always_fire —
  # 청크 끝마다 마지막 단어를 잘라 다음 청크에서 다시 디코딩한다. 한국어(띄어쓰기 단위가 큼)에서
  # "마무 마무리 마무리되었으며" 처럼 조각이 중복돼 CER 0.15~0.2 가 났다(2026-09-23 프로브).
  # --never-fire 는 마지막 단어를 자르지 않는다. 중복이 사라지는지, 지연이 얼마나 느는지 같이 잰다.
  simul-nf) ARGS=(--model large-v3 --backend faster-whisper --backend-policy simulstreaming --never-fire) ;;
  # 원인 규명용 한 점: 청크를 기본 0.1s 에서 1.0s 로. 청크 경계가 10분의 1로 줄면 경계 중복도 그만큼 줄고
  # xRT(기본값 1.10, 실시간을 못 따라감)도 내려가는지 본다. 지연 노브라 기본값 비교 표에는 안 넣는다.
  simul-c1) ARGS=(--model large-v3 --backend faster-whisper --backend-policy simulstreaming --min-chunk-size 1.0) ;;
  local)  ARGS=(--model large-v3 --backend faster-whisper --backend-policy localagreement) ;;
  funasr) ARGS=(--backend funasr --backend-policy localagreement) ;;
  *) echo "unknown combo: $COMBO" >&2; exit 2 ;;
esac
echo "[run_wlk_server] combo=$COMBO port=$PORT  $WLK ${COMMON[*]} ${ARGS[*]} $*"
exec "$WLK" "${COMMON[@]}" "${ARGS[@]}" "$@"
