"""Does a mixed-language slot make the ASR write English in Hangul?

Rebuilds one streaming decode step offline: the slot's audio (from the slot reset to the step)
and the prompt prefix the server gave (`language X<asr_text>` + text already committed in the
slot), then lets the model continue. Conditions change the header, the committed prefix, and
where the slot audio starts.

usage: uv run --project bench python core/research/lid_module/slot_mix.py
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from core.utils import env  # noqa: E402

env.load()

import librosa  # noqa: E402
import torch  # noqa: E402

AUDIO = "/home/mobility/STiTy-team/datasets/northstar-260928/audio/{}.wav"
SR = 16000

# slot start / step end / English onset (s), and the Korean text committed in the slot before it.
CASES = {
    "scene5 (음차됨)": dict(
        scene="scene5", start=11.2, end=21.2, switch=16.1,
        committed="부장님 고향이 부산이세요. <SEG> 그래서 야구만 보면 저렇게 에 사투리가 나오세요. <SEG> ",
    ),
    "scene1 (대조: 맞게 나옴)": dict(
        scene="scene1", start=35.4, end=42.7, switch=None,
        committed="준서야, 뭐라고? <SEG> 아, 아닙니다. 고기는 제가 굽겠습니다. <SEG> ",
    ),
}


def main() -> None:
    from qwen_asr.inference.qwen3_asr import Qwen3ASRModel

    m = Qwen3ASRModel.from_pretrained(
        "Doo12/Qwen3-ASR-1.7B-koen-seg-mix-merged", dtype=torch.float32, device_map="cpu"
    )
    base = m._build_text_prompt(context="", force_language=None)

    def run(wav, prefix):
        m._build_text_prompt = lambda context, force_language: base + prefix
        out = m._infer_asr_transformers([""], [wav], [None])[0]
        return (prefix + out).replace("\n", " ")

    for name, c in CASES.items():
        wav, _ = librosa.load(AUDIO.format(c["scene"]), sr=SR)
        mixed = wav[int(c["start"] * SR) : int(c["end"] * SR)]
        conds = [
            ("혼합 소리 + Korean 머리말 + 앞 한국어 문장 (서버와 같음)", mixed, "language Korean<asr_text>" + c["committed"]),
            ("혼합 소리 + English 머리말 + 앞 한국어 문장", mixed, "language English<asr_text>" + c["committed"]),
            ("혼합 소리 + Korean 머리말, 앞 문장 없음", mixed, "language Korean<asr_text>"),
            ("혼합 소리 + 머리말·앞 문장 없음", mixed, ""),
        ]
        if c["switch"]:
            eng = wav[int(c["switch"] * SR) : int(c["end"] * SR)]
            conds += [
                ("영어 소리만 (전환 때 리셋) + 머리말 없음", eng, ""),
                ("영어 소리만 + Korean 머리말", eng, "language Korean<asr_text>"),
            ]
        print(f"=== {name}", flush=True)
        for label, w, prefix in conds:
            print(f"- {label}\n    {run(w, prefix)}", flush=True)
    print("SLOT_MIX_DONE", flush=True)


if __name__ == "__main__":
    main()
