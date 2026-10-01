#!/bin/bash
# Run the `*_punct-noctx` pipelines on FLEURS en->ko and ko->en (quiet clips raised
# to -23 dB RMS). Per backend: start its server -> 10-sentence run per language ->
# gate -> full run -> stop server. Backends run one at a time: each needs the card
# with the translator. See tools/README.md.
#
#   tmux new-session -d -s backends -c <repo> #     "bash bench/tools/backends_chain.sh qwen qwen-ko qwen-en nemotron wlk voxtral > logs/chain_backends.log 2>&1"
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
REPO=$(cd "$(dirname "$0")/../.." && pwd)
SRV=$REPO/bench/tools/servers
CONDA=$HOME/miniforge3/bin/conda
LOGS=$REPO/logs
mkdir -p $LOGS
cd $REPO
say() { echo "[$(date '+%F %T')] $*"; }

say "start; GPU used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

port_up() { python3 -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1',$1))==0 else 1)"; }
wait_port() {  # wait_port <port> <timeout_sec>
  local i
  for i in $(seq 1 $(( $2 / 5 ))); do port_up $1 && return 0; sleep 5; done
  return 1
}
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; }

SERVER_PID=""
start_server() {  # start_server <backend> <lang>
  local log=$LOGS/server.$1.$2.log
  local p; case $1 in nemotron) p=8766 ;; wlk) p=8791 ;; voxtral) p=8010 ;; esac
  if port_up $p; then say "port $p is already taken by someone else; not using it"; return 1; fi
  case $1 in
    nemotron)
      setsid $CONDA run --no-capture-output -n asr-nemotron python $SRV/server.py \
        --backend nemotron --port 8766 > $log 2>&1 &
      SERVER_PID=$!; PORT=8766; WAIT=600 ;;
    wlk)
      setsid bash $SRV/run_wlk_server.sh local 8791 --lan $2 > $log 2>&1 &
      SERVER_PID=$!; PORT=8791; WAIT=900 ;;
    voxtral)
      VLLM_USE_FLASHINFER_SAMPLER=0 LD_LIBRARY_PATH=$HOME/miniforge3/envs/asr-voxtral/lib \
      setsid $CONDA run --no-capture-output -n asr-voxtral vllm serve \
        mistralai/Voxtral-Mini-4B-Realtime-2602 --tokenizer-mode mistral \
        --gpu-memory-utilization 0.46 --max-model-len 1024 \
        --compilation_config '{"cudagraph_mode":"PIECEWISE"}' --port 8010 > $log 2>&1 &
      SERVER_PID=$!; PORT=8010; WAIT=1200 ;;
  esac
  wait_port $PORT $WAIT || { say "server $1 did not come up; tail:"; tail -20 $log; return 1; }
  if [[ $1 == nemotron ]]; then
    # It loads its model on the first connection; do that before bench connects.
    $REPO/bench/.venv/bin/python -c "import asyncio, websockets
async def w():
    async with websockets.connect(\"ws://127.0.0.1:8766\", open_timeout=900) as ws:
        print(\"warmup:\", (await asyncio.wait_for(ws.recv(), 900))[:80])
asyncio.run(w())" || return 1
  fi
  say "server $1 ($2) up on :$PORT, GPU used $(gpu_used) MiB"
}
stop_server() {
  [[ -n "$SERVER_PID" ]] && kill -TERM -- -$SERVER_PID 2>/dev/null
  sleep 10
  [[ -n "$SERVER_PID" ]] && kill -KILL -- -$SERVER_PID 2>/dev/null
  SERVER_PID=""
  sleep 5
  say "server stopped, GPU used $(gpu_used) MiB"
}
trap stop_server EXIT

run() {  # run <backend> <dataset> -> rc
  local cfg=$(cfg_of $1)
  say "RUN $1 $2"
  make bench CONFIG=$cfg DATASET=$2 > $LOGS/$2.$cfg.log 2>&1
  local rc=$?
  say "END $1 $2 rc=$rc"
  return $rc
}
cfg_of() {
  case $1 in
    qwen)     echo asr.qwen-seg-base+mt.qwen3.5-4b_punct-noctx ;;
    qwen-ko)  echo asr.qwen-seg-ko+mt.qwen3.5-4b_punct-noctx ;;
    qwen-en)  echo asr.qwen-seg-en+mt.qwen3.5-4b_punct-noctx ;;
    nemotron) echo asr.nemotron-0.6b+mt.qwen3.5-4b_punct-noctx ;;
    wlk)      echo asr.wlk-local+mt.qwen3.5-4b_punct-noctx ;;
    voxtral)  echo asr.voxtral-mini-4b+mt.qwen3.5-4b_punct-noctx ;;
  esac
}
gate() {  # gate <backend> <dataset>: sanity of a 10-sentence run
  python3 - "$REPO/bench/runs/$2/$(cfg_of $1)" <<'EOF'
import json, sys
d = sys.argv[1]
try:
    s = json.load(open(f"{d}/summary.json"))
except Exception as e:
    print(f"GATE FAIL no summary: {e}"); sys.exit(1)
rows = [json.loads(l) for l in open(f"{d}/items.jsonl")]
m, c = s["metrics"], s["counts"]
ko = rows[0].get("src_lang") == "ko"
err = m.get("cer") if ko else m.get("wer")
split = (lambda t: list("".join((t or "").split()))) if ko else (lambda t: (t or "").split())
ref = sum(len(split(r["reference"])) for r in rows)
hyp = sum(len(split(r["transcription_output"])) for r in rows)
ratio = hyp / max(ref, 1)
bad = []
if c["errored"]: bad.append(f"errored={c['errored']}")
if c["empty_transcription_output"]: bad.append(f"empty={c['empty_transcription_output']}")
if err is None or err > 0.30: bad.append(f"{'cer' if ko else 'wer'}={err}")
if not 0.8 <= ratio <= 1.25: bad.append(f"hyp/ref length={ratio:.2f}")
print(f"GATE {'FAIL ' + ', '.join(bad) if bad else 'PASS'}  "
      f"{'cer' if ko else 'wer'}={err:.3f} hyp/ref={ratio:.2f} "
      f"commits={ {k: round(v, 2) for k, v in m['commit_reasons'].items() if v} }")
sys.exit(1 if bad else 0)
EOF
}

for B in "${@:-qwen}"; do
  for L in en ko; do
    [[ $L == en ]] && PAIR=en-ko_vol23 || PAIR=ko-en_vol23
    # Qwen runs inside bench. WLK takes its language when the server starts;
    # the others take it per session.
    if [[ $B != qwen* ]] && [[ $B == wlk || -z "$SERVER_PID" ]]; then
      [[ -n "$SERVER_PID" ]] && stop_server
      start_server $B $L || { say "SKIP $B: server failed"; touch $LOGS/backends.$B.server-failed; break; }
    fi
    DS=fleurs_${PAIR/_vol23/}_vol23_first10
    run $B $DS
    if gate $B $DS; then
      run $B fleurs_$PAIR && touch $LOGS/backends.$B.$PAIR.done
    else
      say "SKIP full $B $PAIR: 10-sentence check failed"
      touch $LOGS/backends.$B.$PAIR.gate-failed
    fi
  done
  stop_server
done
say "CHAIN_DONE"
