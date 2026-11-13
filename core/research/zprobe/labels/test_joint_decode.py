"""Joint transcript+translation adapter (A1, prefix16jt): the `dT<blank2>dX` split and the zprobe-prefix `joint: true`
state handling. Pure CPU; the component tests use a scripted fake model and the local Qwen3-ASR tokenizer (skipped
when the model dir is absent)."""

import asyncio
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from core.components.transcription.zprobe_prefix import (  # noqa: E402
    JOINT_SRC, ZprobePrefixTranscription, split_joint)

TOK_DIR = os.path.join(ROOT, "models", "Qwen3-ASR-1.7B-koen-seg-mix-merged")


@pytest.mark.parametrize("txt,want", [
    ("x<blank2>y", ("x", "y")),
    (" anniversary<blank2> 사찰 기념일의", (" anniversary", " 사찰 기념일의")),
    ("x<blank2>", ("x", "")),
    ("<blank2>y", ("", "y")),
    (" <SEG>", ("", " <SEG>")),
    ("", ("", "")),
    ("partial garbage", ("partial garbage", "")),
    ("x<blank2>y<blank2>z", ("x", "yz")),
    (" end.<SEG><blank2> 끝.", (" end.", "<SEG> 끝.")),
    ("a<blank1>b<blank2>c", ("ab", "c")),
])
def test_split_joint(txt, want):
    assert split_joint(txt) == want


# ---- component with a scripted fake model ---------------------------------------------------------------

class _Batch(dict):
    def to(self, *_):
        return self


class _FakeProcessor:
    def __init__(self, tok, log):
        self.tokenizer = tok
        self._log = log

    def __call__(self, text, audio, return_tensors, padding):
        import torch

        self._log.append(list(text))
        ids = [self.tokenizer.encode(t, add_special_tokens=False) for t in text]
        n = max(map(len, ids))
        pad = self.tokenizer.pad_token_id or 0
        return _Batch(input_ids=torch.tensor([[pad] * (n - len(x)) + x for x in ids]))


class _FakeInner:
    def __init__(self, tok, script):
        import torch

        self._tok, self._script = tok, script
        self.thinker = torch.nn.Linear(1, 1)

    def generate(self, input_ids, **_):
        import torch

        outs = []
        for _row in input_ids:
            cont = self._tok.encode(self._script.pop(0), add_special_tokens=False) + [151645]
            outs.append(cont)
        n = max(map(len, outs))
        new = torch.tensor([o + [151643] * (n - len(o)) for o in outs])
        return torch.cat([input_ids, new], dim=1)


class _FakeModel:
    def __init__(self, tok, script, log):
        self.processor = _FakeProcessor(tok, log)
        self.model = _FakeInner(tok, script)

    def _build_text_prompt(self, context, force_language):
        return "<|im_start|>system\n" + context + "<|im_end|>\n<|im_start|>assistant\n"


def _component(script, **settings):
    from transformers import AutoTokenizer

    if not os.path.isdir(TOK_DIR):
        pytest.skip("Qwen3-ASR tokenizer not present")
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    log = []
    c = ZprobePrefixTranscription({"targets": ["ko"], "step_sec": 0.5, "first_sec": 0.5, **settings}, cfg=None)
    c.model = _FakeModel(tok, list(script), log)
    c._lid = None
    c.start()
    return c, log


def _feed(c, sec=0.5):
    return asyncio.run(c.transcribe((np.zeros(int(16000 * sec), dtype=np.int16)).tobytes()))


def test_joint_off_prompt_and_state_unchanged():
    c, log = _component([" 대부분의", " 사찰"])
    out1, out2 = _feed(c), _feed(c)
    base = c._bases["ko"]
    assert log == [[base], [base + " 대부분의"]]
    assert [o.translation for o in out1 + out2] == ["대부분의", "사찰"]
    assert c._raw["ko"] == " 대부분의 사찰" and c._raw_t["ko"] == "" and c.joint_transcript == []


def test_joint_on_prompt_split_and_state():
    c, log = _component(["Most of<blank2> 대부분의", " the temple<blank2>", "<blank2> 사찰"], joint=True)
    outs = _feed(c) + _feed(c) + _feed(c)
    base = c._bases["ko"]
    assert log == [[base + JOINT_SRC], [base + " 대부분의" + JOINT_SRC + "Most of"],
                   [base + " 대부분의" + JOINT_SRC + "Most of the temple"]]
    assert [o.translation for o in outs] == ["대부분의", "사찰"]
    assert c._raw["ko"] == " 대부분의 사찰" and c._raw_t["ko"] == "Most of the temple"
    assert [d for _, _, d in c.joint_transcript] == ["Most of", " the temple"]


def test_joint_latin_block_skips_both():
    c, _ = _component(["Hello<blank2> hello world"], joint=True, latin_commit_guard=True)
    assert _feed(c) == []
    assert c._raw["ko"] == "" and c._raw_t["ko"] == "" and c.joint_transcript == []


def test_joint_seg_pair_row_resets():
    c, _ = _component(["Most of<blank2> 대부분의", " <SEG>"], joint=True, seg_reset=True, seg_keep_sec=0.0)
    _feed(c)
    assert c._raw_t["ko"] == "Most of"
    _feed(c)
    assert c._raw["ko"] == "" and c._raw_t["ko"] == ""
