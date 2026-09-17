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

from collections import defaultdict

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = "google/madlad400-3b-mt"
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_CODE = {"en": "eng_Latn", "de": "deu_Latn", "ja": "jpn_Jpan",
             "zh": "zho_Hans", "ko": "kor_Hang", "es": "spa_Latn"}


def hit_cap(seq, n_prefix: int, max_new: int, stop_ids: set,
            budget: int | None = None) -> bool:
    """이 행이 `max_new` 에 걸려 끊겼는가 — 끊겼으면 상한 없이 다시 돌려야 한다.

    끝까지 간 행은 마지막이 eos 이고, 먼저 끝난 행은 그 뒤가 pad 다. 둘 다 없이 `max_new`
    개를 채웠다면 상한이 문 것이다. `budget`(= `max_new_tokens`) 과 같은 상한은 상한 없는
    경우 그 자체라 다시 돌릴 것이 없다.
    """
    if budget is not None and max_new >= budget:
        return False
    gen = [int(t) for t in seq[n_prefix:n_prefix + max_new]]
    return len(gen) == max_new and not any(t in stop_ids for t in gen)


class Nmt:
    def __init__(self, src: str = "en", tgt: str = "de", device: str = "cuda",
                 model_name: str = MODEL, max_new_tokens: int = 128,
                 attentions: bool = False, attn_layer: int | None = None,
                 cap_tokens: bool = True):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.cap_tokens = cap_tokens
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

    # ── 배치 ────────────────────────────────────────────────────────────────
    # 배치 1 디코드는 스텝마다 디코더 가중치(3B fp16 의 절반쯤)를 통째로 읽고 그 한 줄에만
    # 쓴다. GB10 처럼 통합메모리 대역폭(~273GB/s)이 병목인 기계에서는 이게 곧 속도다.
    # 여러 문장의 같은 회차를 한 배치로 묶으면 그 읽기를 배치 전체가 나눠 쓴다.

    def _encode_batch(self, texts: list[str]):
        if not self.is_nllb:
            texts = [f"<2{self.tgt_code}> {t}" for t in texts]
        return self.tok(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=256).to(self.device)

    def _cap_new(self, texts: list[str]) -> int:
        """이 묶음에 쓸 `max_new_tokens`.

        **배치는 모든 행이 EOS 를 낼 때까지 돈다.** 한 행이 안 멈추면 이미 끝난 나머지도
        같이 돈다. en→ja 는 madlad 가 같은 말을 되풀이하며 EOS 를 안 내는 행이 섞이고,
        그 한 행이 배치 전체를 128 스텝까지 끌고 갔다 — 6어절 접두사 128문장 실측에서
        배치스텝 128 에 낭비 90% 였다 (de 16 스텝 36%, zh 18 스텝 44%). ja 가 de 의
        3배 느렸던 원인이 이것이고, 접두사가 길수록 폭주 행이 늘어 회차가 갈수록 나빠진다
        (8어절 1/128, 20어절 2/128).

        상한은 소스 어절수로 잡는다. 접두사 어절수별 출력 길이 실측에서 **정상 분포의 위쪽과
        폭주 사이가 비어 있다** — ja 20어절이 p90 40 인데 그 위는 곧바로 128 이고, 그 사이
        값이 없다. 그래서 `4×어절 + 16` 으로 자르면 정상 번역은 못 건드리고 폭주만 끊긴다.
        측정한 모든 점에서 정상 최대 대비 33% 이상 여유가 있다 (ja 8어절 최대 36 vs 상한 48,
        de 20어절 최대 52 vs 96).

        묶음 안에 어절수가 다른 행이 섞이므로 **최댓값**을 쓴다 — 짧은 행이 손해 보지 않는다.
        `forced` 가 번역의 앞부분을 이미 깔고 있으면 남겨야 할 새 토큰은 더 적으므로, 접두사
        전체 길이로 잡는 이 상한은 그만큼 더 넉넉하다.
        """
        if not self.cap_tokens:
            return self.max_new_tokens
        w = max((len(t.split()) for t in texts), default=1)
        return min(self.max_new_tokens, 4 * w + 16)

    @property
    def _stop_ids(self) -> set:
        return {i for i in (self.eos_id, self.tok.pad_token_id) if i is not None}

    def _uncapped(self, fn, *a):
        """상한을 끄고 한 번 부른다 — 상한에 닿은 행을 다시 돌릴 때만 쓴다."""
        old, self.cap_tokens = self.cap_tokens, False
        try:
            return fn(*a)
        finally:
            self.cap_tokens = old

    def _decoder_prefix_batch(self, forced_rows: list[list[int] | None]) -> torch.Tensor:
        """행마다 **자기** forced 를 깐다.

        길이가 같다고 내용이 같은 것이 아니다 — 한 행의 forced 를 배치 전체에 복제하면
        다른 문장의 확정 번역 위에서 이어 디코딩하게 되고, 결과가 조용히 틀린다
        (실측: mu_prefix 평균 조각수 4.70 → 2.53).
        """
        head = [self.start_id] + ([self.tgt_id] if self.is_nllb else [])
        return torch.tensor([head + list(f or []) for f in forced_rows], device=self.device)

    @staticmethod
    def _by_prefix_len(items, max_batch: int):
        """`(소스, forced)` 목록을 **forced 길이가 같은 것끼리** 묶어 인덱스로 낸다.

        **디코더 접두사는 패딩하면 안 된다.** 좌패딩 + `decoder_attention_mask` 로 길이를
        맞추면 패드 자리의 self-attention 이 전부 마스크돼 softmax 가 NaN 이 되고, 실제
        토큰이 그 자리를 0 가중치로 곱해도 NaN 은 남아 배치가 통째로 망가진다 (실측:
        패딩된 행이 `decoder_start` 만 반복 출력하고 교차어텐션이 NaN). 길이가 같은
        것끼리만 묶으면 패딩이 아예 없고, 4문장 실측에서 생성 토큰과 교차어텐션 argmax 가
        단건과 정확히 일치했다. 인코더 쪽 우패딩은 무해하다 — 질의마다 실제 키가 있어
        NaN 이 안 나고, 어절 매핑은 행별 실제 길이로 자른다.
        """
        g: dict[int, list[int]] = defaultdict(list)
        for i, it in enumerate(items):
            g[len(it[1] or [])].append(i)
        for idxs in g.values():
            for s in range(0, len(idxs), max_batch):
                yield idxs[s:s + max_batch]

    @torch.inference_mode()
    def emit_with_alignment_batch(self, items: list[tuple[str, list[int] | None]],
                                  max_batch: int = 32) -> list[list[tuple[int, int]]]:
        """`emit_with_alignment` 의 배치판. 반환 순서는 입력 순서다.

        **상한에 닿은 행은 상한 없이 다시 돌린다.** AlignAtt 은 여기서 받은 토큰을 그대로
        `forced` 로 커밋하므로, 폭주 행을 자른 위치가 이후 회차를 전부 바꾼다. 다시 도는 행은
        ja 긴문장 기준 1% 미만이라 상한이 벌어 준 속도는 거의 그대로 남는다.
        """
        res: list[list[tuple[int, int]]] = [[] for _ in items]
        retry: list[int] = []
        for idxs in self._by_prefix_len(items, max_batch):
            texts = [items[i][0] for i in idxs]
            enc = self._encode_batch(texts)
            dec = self._decoder_prefix_batch([items[i][1] for i in idxs])
            cap = self._cap_new(texts)
            out = self.model.generate(
                **enc, decoder_input_ids=dec,
                num_beams=1, do_sample=False, max_new_tokens=cap,
                output_attentions=True, return_dict_in_generate=True)
            n_prefix = dec.shape[1]
            real = enc["attention_mask"].sum(1).tolist()
            # 32층을 다 들고 있으면 메모리가 커진다. 쓰는 층만 스텝별로 꺼내 둔다.
            att = [out.cross_attentions[s][self.attn_layer].float().mean(1)[:, -1]
                   for s in range(len(out.cross_attentions))]
            for b, i in enumerate(idxs):
                w_of = self._word_of_token(enc["input_ids"][b][:real[b]])
                n_src = len(w_of)
                lo = next((k for k, x in enumerate(w_of) if x >= 0), 1)
                seq = out.sequences[b]
                if hit_cap(seq, n_prefix, cap, self._stop_ids, self.max_new_tokens):
                    retry.append(i)
                    continue
                for step in range(len(att)):
                    if n_prefix + step >= len(seq):
                        break
                    tok_id = int(seq[n_prefix + step])
                    if tok_id in (self.eos_id, self.tok.pad_token_id):
                        break
                    a = att[step][b][:n_src]
                    j = int(a[lo:n_src - 1].argmax()) + lo
                    res[i].append((tok_id, w_of[j]))
        if retry:
            sub = self._uncapped(self.emit_with_alignment_batch,
                                 [items[i] for i in retry], max_batch)
            for k, i in enumerate(retry):
                res[i] = sub[k]
        return res

    @torch.inference_mode()
    def translate_prefix_batch(self, items: list[tuple[str, list[int] | None]],
                               max_batch: int = 64) -> list[tuple[str, list[int]]]:
        """`translate_prefix` 의 배치판. 반환 순서는 입력 순서다.

        상한에 닿은 행은 상한 없이 다시 돌린다 — `emit_with_alignment_batch` 와 같은 이유다.
        """
        res: list[tuple[str, list[int]]] = [("", [])] * len(items)
        retry: list[int] = []
        for idxs in self._by_prefix_len(items, max_batch):
            texts = [items[i][0] for i in idxs]
            enc = self._encode_batch(texts)
            dec = self._decoder_prefix_batch([items[i][1] for i in idxs])
            cap = self._cap_new(texts)
            out = self.model.generate(
                **enc, decoder_input_ids=dec,
                num_beams=1, do_sample=False, max_new_tokens=cap)
            for b, i in enumerate(idxs):
                if hit_cap(out[b], dec.shape[1], cap, self._stop_ids, self.max_new_tokens):
                    retry.append(i)
                    continue
                ids = self._strip(out[b])
                res[i] = (self.tok.decode(ids, skip_special_tokens=True), ids)
        if retry:
            sub = self._uncapped(self.translate_prefix_batch,
                                 [items[i] for i in retry], max_batch)
            for k, i in enumerate(retry):
                res[i] = sub[k]
        return res

    @torch.inference_mode()
    def full_candidates_batch(self, texts: list[str], n: int = 10,
                              max_beams: int = 128) -> list[list[str]]:
        """`full_candidates` 의 배치판.

        빔이 배치 안에서 곱해지므로 예산은 문장 수가 아니라 **문장 수 × n** 으로 잡는다.
        n=50 이면 한 번에 두 문장뿐이다."""
        res: list[list[str]] = []
        retry: list[int] = []
        step = max(1, max_beams // max(n, 1))
        for s in range(0, len(texts), step):
            chunk = texts[s:s + step]
            enc = self._encode_batch(chunk)
            dec = self._decoder_prefix_batch([None] * len(chunk))
            cap = self._cap_new(chunk)
            out = self.model.generate(
                **enc, decoder_input_ids=dec,
                num_beams=n, num_return_sequences=n,
                max_new_tokens=cap, do_sample=False)
            for b in range(len(chunk)):
                rows = out[b * n:(b + 1) * n]
                # 후보 하나라도 상한에 닿았으면 그 문장을 통째로 다시 돌린다 — 후보 순위가
                # 잘린 후보에 걸려 바뀔 수 있다.
                if any(hit_cap(x, dec.shape[1], cap, self._stop_ids, self.max_new_tokens)
                       for x in rows):
                    retry.append(s + b)
                res.append([self.tok.decode(self._strip(x), skip_special_tokens=True)
                            for x in rows])
        if retry:
            sub = self._uncapped(self.full_candidates_batch,
                                 [texts[i] for i in retry], n, max_beams)
            for k, i in enumerate(retry):
                res[i] = sub[k]
        return res
