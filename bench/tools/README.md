# bench/tools/ — Qwen 이 아닌 스트리밍 ASR 을 bench 로 재기

Voxtral·Nemotron·WhisperLiveKit(WLK)은 의존성이 bench 와 맞지 않아 각자 conda env 의 서버로 띄우고,
bench 의 `remote-stream` 부품이 그 서버에 붙는다([../README.md](../README.md) 의 설정 절). 이 디렉터리에는
그 서버들과, 백엔드 여럿을 차례로 돌리는 체인이 있다.

```bash
tmux new-session -d -s backends -c <repo> \
  "bash bench/tools/backends_chain.sh qwen qwen-ko qwen-en nemotron wlk voxtral > logs/chain_backends.log 2>&1"
```

**백엔드는 한 번에 하나다.** 번역기(Qwen3.5-4B)와 ASR 이 한 카드(4090, 24 GB)를 같이 쓰므로 둘이 동시에
뜨면 넘친다. 체인은 백엔드마다 서버를 띄우고, 언어마다 10문장(`*_vol23_first10`)을 먼저 돌려 아래 관문을
넘을 때만 전체(`*_vol23`)를 돌린 뒤 서버를 내린다. 관문: 오류·빈 전사 0, WER/CER 0.30 이하, 전사 길이가
정답의 0.8~1.25배. 마지막 조건은 경계 단어가 두 번씩 커밋되는 실수를 잡는다.

체인은 PID 가 아니라 로그로 순서를 잡는다. 체인을 이어 붙일 때 앞 체인의 끝을 기다리려면
`grep -qE "\] CHAIN_DONE$"` 처럼 **줄 끝까지** 맞춘다 — 대기 안내 문구에도 `CHAIN_DONE` 이라는 글자가
들어 있어서 그냥 `grep CHAIN_DONE` 으로는 바로 통과해 버린다.

## servers/

| 백엔드 | 띄우는 법 | 포트 | 언어 힌트 |
|---|---|---|---|
| Nemotron | `asr-nemotron` env 에서 `servers/server.py --backend nemotron` | 8766 | 세션마다 `start{lang}` |
| WLK | `servers/run_wlk_server.sh local <port> --lan <ko\|en>` (`asr-wlk` env) | 8791 | 서버를 띄울 때 `--lan`. 언어가 바뀌면 다시 띄운다 |
| Voxtral | `asr-voxtral` env 에서 `vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602` | 8010 | 없다. realtime API 에 칸이 없다 |

`server.py`·`protocol.py`·`engine_nemotron.py` 는 `feat/multi-asr-backends` 의 `evaluation/backends/` 에서,
`run_wlk_server.sh` 는 `bench/wlk` 의 같은 자리에서 가져왔다. 바꾼 것은 Nemotron 하나다(아래).

### Nemotron — 붙잡은 마지막 단어를 `pending` 으로 보낸다

엔진은 다음 단어가 시작돼야 앞 단어가 끝났다고 보고 내보낸다(단어 중간에서 끊어 `lag be hind` 가 되는 것을
막으려고). 그래서 문장의 마지막 단어는 다음 말이 나올 때까지 서버에 남는다. 원본 그대로 붙이면 그 단어가
VAD 커밋에서 빠지고 스트림이 닫힐 때 혼자 커밋된다 — FLEURS en→ko 커밋의 45%가 그랬다.

`engine_nemotron.py` 에 `pending()` 을 더해 붙잡은 단어를 돌려주게 했고, `protocol.py` 는 그 단어를
`final` 의 `pending` 필드나 `partial{text}` 로 함께 보낸다. `remote-stream` 은 이것을 고칠 수 있는 꼬리 글로
다루어 VAD 커밋 때 함께 넣는다. 스트림 끝 커밋이 1% 로 줄었다.

모델은 **첫 연결 때** 불러온다. 그 사이 다음 연결은 websockets 의 기본 10초 안에 handshake 를 못 끝내
끊긴다. 체인은 서버를 띄운 뒤 한 번 접속해 `hello` 를 받을 때까지 기다리고 나서 bench 를 붙인다.

### Voxtral — 번역기와 한 카드에 겨우 들어간다

가중치가 8.4 GB 다. 원래 설정(`--gpu-memory-utilization 0.85 --max-model-len 16384`)으로는 번역기와 같이
못 뜨고, `0.42 / 4096` 으로도 KV 캐시가 0.17 GB 밖에 안 남아 뜨지 않는다. `0.46 / 1024` 로 띄우고 Voxtral
파이프라인 설정에서만 번역기를 `gpu_memory_utilization: 0.44` 로 낮췄다(번역 결과는 같다). 1024 토큰은
FLEURS 의 가장 긴 문장(29초, 약 420토큰)에는 넉넉하지만 회의·발표를 한 세션으로 흘리면 금방 닿는다.

이 박스에는 nvcc 가 없어 `VLLM_USE_FLASHINFER_SAMPLER=0` 이 필요하다.
