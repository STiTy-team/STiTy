"""Zhang 2020 Algorithm 1 이 요구하는 NMT 능력만 감싼다.

필요한 것 둘 — gtx 로는 어느 쪽도 안 된다:
    1. 강제 디코딩 (tgt_force = 직전까지 확정된 MU 번역)
    2. 전체 문장 번역의 beam top-N 후보 (엄격한 접두사 조건 완화, 논문 N=10)

모델 하나로 de/ja/zh 를 덮는다. 타깃마다 다른 Marian 을 쓰면 모델 품질 차이가 타깃 간
비교를 오염시킨다 (en→ja Marian 은 특히 약하다).

**기본값은 평가 번역기와 같은 madlad 다.** 여기 나오는 규칙들(AlignAtt 의 어텐션, MU 의
접두사 일치)은 전부 "이 모델이 지금 무엇을 아는가" 를 묻는데, 그 모델이 실제로 번역을
내놓는 모델과 다르면 엉뚱한 모델에 대해 판정하는 셈이 된다. MU 는 특히 그렇다 — 접두사
일치는 모델마다 답이 다르다. 원논문들은 각자 자기 시스템 하나로 판정과 출력을 같이 하므로,
그 결합을 맞추려면 `pipeline.LOCAL_MT_DEFAULT` 와 같은 모델이어야 한다.

타깃 지정 규약이 모델마다 다르다. madlad 는 소스 앞에 `<2xx>` 를 붙이고 디코더는 시작
토큰 하나로 시작한다. nllb 는 소스 언어를 토크나이저에, 타깃 언어를 디코더 두 번째
토큰으로 준다. `_lang_prefix_len` 이 그 차이를 흡수한다.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = "google/madlad400-3b-mt"
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_CODE = {"en": "eng_Latn", "de": "deu_Latn", "ja": "jpn_Jpan",
             "zh": "zho_Hans", "ko": "kor_Hang", "es": "spa_Latn"}


class Nmt:
    def __init__(self, src: str = "en", tgt: str = "de", device: str = "cuda",
                 model_name: str = MODEL, max_new_tokens: int = 128,
                 attentions: bool = False, attn_layer: int | None = None):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.is_nllb = "nllb" in model_name.lower()
        self.tgt_code = tgt
        tok_kw = {"src_lang": NLLB_CODE[src]} if self.is_nllb else {}
        self.tok = AutoTokenizer.from_pretrained(model_name, **tok_kw)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        # 교차어텐션을 뽑으려면 sdpa 로는 안 되고 eager 여야 한다.
        kw = {"attn_implementation": "eager"} if attentions else {}
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, dtype=dtype, **kw).to(device).eval()
        self.tgt_id = self.tok.convert_tokens_to_ids(NLLB_CODE[tgt]) if self.is_nllb else None
        self.start_id = self.model.config.decoder_start_token_id
        self.eos_id = self.tok.eos_token_id
        # 디코더 앞머리 길이 — nllb 는 [start, 타깃언어], madlad 는 [start] 뿐이다.
        self._lang_prefix_len = 2 if self.is_nllb else 1
        # 소스 앞머리에서 어절이 아닌 토큰 수 (nllb 는 소스 언어 코드 1개,
        # madlad 는 `<2xx>` 가 `['▁', '<2xx>']` 로 2개).
        self._tag_len = 1 if self.is_nllb else len(
            self.tok(f"<2{tgt}>", add_special_tokens=False)["input_ids"])
        # 층은 모델마다 다시 골라야 한다. 기본값은 `probe_attn_layer.py` 의 정렬 단조성
        # 스윕에서 **타깃 평균이 가장 높은** 층이다 — nllb-600M(12층) 5, madlad-3B(32층) 26.
        # madlad 30문장 실측(de/ja/zh): 26 이 0.8009 로 최고, 30 이 0.8008 로 사실상 동률.
        # **타깃마다 최고가 다르다** (de 는 4 에서 0.854, zh 는 26 에서 0.853). ja 는 0 층이
        # 0.855 로 가장 높게 나오지만 이건 신호가 아니라 잡음으로 본다 — 첫 층의 교차어텐션은
        # 아직 퍼져 있어 argmax 가 왼쪽에서 오른쪽으로 천천히 끌려가고, 그러면 내용을 하나도
        # 안 봐도 단조성이 높게 나온다. 같은 층이 de·zh 에서는 꼴찌다(0.734 / 0.753).
        self.attn_layer = attn_layer if attn_layer is not None else (5 if self.is_nllb else 26)

    def _encode(self, text: str):
        # madlad 는 타깃을 소스 앞의 `<2xx>` 로 받는다. nllb 는 토크나이저의 src_lang 으로 받았다.
        if not self.is_nllb:
            text = f"<2{self.tgt_code}> {text}"
        return self.tok(text, return_tensors="pt", truncation=True,
                        max_length=256).to(self.device)

    def _decoder_prefix(self, forced: list[int] | None) -> torch.Tensor:
        ids = [self.start_id] + ([self.tgt_id] if self.is_nllb else []) + list(forced or [])
        return torch.tensor([ids], device=self.device)

    def _strip(self, seq: torch.Tensor) -> list[int]:
        """[start, (언어,) …, eos] → 가운데 본문 토큰만."""
        n = self._lang_prefix_len
        out = [int(t) for t in seq]
        out = out[n:] if len(out) >= n else out
        while out and out[-1] in (self.eos_id, self.tok.pad_token_id):
            out.pop()
        return out

    @torch.inference_mode()
    def full_candidates(self, text: str, n: int = 10) -> list[str]:
        """전체 문장 번역 상위 N 후보. 첫 번째가 beam 1위다."""
        enc = self._encode(text)
        out = self.model.generate(
            **enc, decoder_input_ids=self._decoder_prefix(None),
            num_beams=n, num_return_sequences=n,
            max_new_tokens=self.max_new_tokens, do_sample=False)
        return [self.tok.decode(self._strip(s), skip_special_tokens=True) for s in out]

    @torch.inference_mode()
    def translate_prefix(self, text: str,
                         forced: list[int] | None = None) -> tuple[str, list[int]]:
        """소스 접두사를 강제 타깃 접두사 위에서 이어 디코딩한다 (greedy, 결정론적)."""
        enc = self._encode(text)
        out = self.model.generate(
            **enc, decoder_input_ids=self._decoder_prefix(forced),
            num_beams=1, do_sample=False, max_new_tokens=self.max_new_tokens)
        ids = self._strip(out[0])
        return self.tok.decode(ids, skip_special_tokens=True), ids

    def _word_of_token(self, ids: torch.Tensor) -> list[int]:
        """소스 토큰 인덱스 → 어절 인덱스. `▁` 로 시작하면 새 어절이다.

        맨 앞의 언어 표지와 마지막 `</s>` 는 어절이 없다 (−1). **madlad 의 `<2de>` 는
        `['▁', '<2de>']` 두 토큰이고 첫 토큰이 `▁` 로 시작한다** — `▁` 규칙에만 맡기면
        표지가 어절 0번이 되어 전체가 한 칸씩 밀리고, 어텐션이 표지로 쏠릴 때(실측: 전
        토큰이 0번) 정렬이 통째로 무의미해진다. 그래서 표지를 토크나이즈해 그 길이만큼
        건너뛴다."""
        toks = self.tok.convert_ids_to_tokens(ids)
        out, w = [], -1
        for i, t in enumerate(toks):
            if i < self._tag_len or t == self.tok.eos_token or t == "</s>":
                out.append(-1)
                continue
            if t.startswith("\u2581"):
                w += 1
            out.append(max(w, 0))
        return out

    @torch.inference_mode()
    def emit_with_alignment(self, text: str, forced: list[int] | None = None
                            ) -> list[tuple[int, int]]:
        """소스 접두사를 디코딩하며 **새 토큰마다 정렬된 소스 어절**을 함께 낸다.

        AlignAtt 이 요구하는 것이 이것뿐이다 — 토큰 i 의 교차어텐션 argmax 가 어느
        소스 어절을 가리키는가. 마지막 층은 `</s>` 로 쏠려(attention sink) 못 쓰므로
        `attn_layer`(기본 5 — 50문장 스윕에서 de 0.829 / ja 0.735 로 합산 최고)를
        쓰고 헤드는 평균낸다. 원 논문은 6층 중 4층 + 헤드 평균이다(아키텍처가 달라 층
        번호는 그대로 옮길 수 없다). 층 간 단조성 차이는 de 0.72~0.84 로 작다.
        """
        enc = self._encode(text)
        out = self.model.generate(
            **enc, decoder_input_ids=self._decoder_prefix(forced),
            num_beams=1, do_sample=False, max_new_tokens=self.max_new_tokens,
            output_attentions=True, return_dict_in_generate=True)
        w_of = self._word_of_token(enc["input_ids"][0])
        n_src = len(w_of)
        seq = out.sequences[0]
        n_prefix = self._lang_prefix_len + len(forced or [])
        new_ids = [int(t) for t in seq[n_prefix:]]
        res: list[tuple[int, int]] = []
        for step, tok_id in enumerate(new_ids):
            if step >= len(out.cross_attentions):
                break
            if tok_id in (self.eos_id, self.tok.pad_token_id):
                break
            a = out.cross_attentions[step][self.attn_layer].float().mean(1)[0, -1]
            # 언어 표지와 </s> 는 정렬 후보에서 뺀다 (어절이 −1 인 자리).
            lo = next((k for k, x in enumerate(w_of) if x >= 0), 1)
            j = int(a[lo:n_src - 1].argmax()) + lo
            res.append((tok_id, w_of[j]))
        return res
