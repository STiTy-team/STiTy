"""번역 문면과 카드 추출 문면.

번역 문면은 1.1(LongContextMT)의 SYSTEM 에 <notes> 안내 한 문장만 더한다. 카드가 아직 없는 문장은 1.1 의 N=0 과
같은 문면이 된다. 조건마다 카드의 어느 칸을 보여 줄지만 다르다.

K1 요약   K2 용어·인물   K3 상황(장르·화자·청중·말투)   K4 전부
K2b 용어를 고유명사로만 제한. 추출기는 새 항목만 내고, 합치기는 코드가 한다 — 30개가 차면 원문에 가장 오래
    안 나온 것부터 뺀다 (K2 는 일반 단어가 섞이고 강연 중반에 30개가 차서 뒤 이름이 못 들어갔다)
    V2 추출(K2b·K3b)은 바뀐 칸만 낸다 — 요약·상황은 바뀌지 않으면 null, 용어는 새 항목만. 출력 토큰을 줄이려는 것
    (K1~K4 는 매번 카드 전체를 다시 써 추출 한 번에 출력이 약 520토큰이었다)
K5  효과가 있던 칸을 합친다: 요약(K1) + 고유명사(K2b 방식) + 말투를 포함한 상황(K3). 추출은 V2 방식(바뀐 칸만)에
    상황의 말투 칸만 다시 허용한다
K3b 상황에서 말투 칸을 뺀다. 장르·화자·청중만 주고 말투는 번역기가 문장마다 정한다 (K3 은 추출기가 매번
    합니다체로 정해 정답 74% 보다 많은 90% 가 합니다체가 됐다)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "LongContextMT" / "scripts"))

from lcmt.prompt import SYSTEM  # noqa: E402

FIELDS = {"K1": ("summary",), "K2": ("terms",), "K3": ("situation",), "K4": ("summary", "terms", "situation"),
          "K2b": ("terms",), "K3b": ("situation",), "K5": ("summary", "terms", "situation")}
V2 = ("K2b", "K3b", "K5")
WITH_REGISTER = ("K5",)
MAX_TERMS = 30

NOTES_SYSTEM = SYSTEM + (" Background notes about the talk so far may be given inside <notes>. An assistant wrote "
                         "them and they may be incomplete. Use them only to keep terms, names, style and politeness "
                         "level consistent and to understand the CURRENT sentence; never translate or repeat them.")

EXTRACT_SYSTEM = (
    "You keep background notes for a live English-to-Korean interpreter of a talk. You get the current notes as "
    "JSON and the next sentences of the talk, each with the Korean translation already shown to the audience. "
    "Return the updated notes as JSON with exactly these keys and nothing else:\n"
    "\"summary\": at most 3 English sentences on what the talk is about and where it is now. Keep what is still "
    "true and change only what the new sentences change.\n"
    "\"terms\": a list of {\"en\": ..., \"ko\": ...} for names of people, places and organisations and for "
    "recurring key terms, with the Korean rendering used in the translations. Keep existing entries as they are "
    "(the audience has already seen them) and add new ones. At most 30 entries.\n"
    "\"situation\": {\"genre\": ..., \"speaker\": ..., \"audience\": ..., \"register\": ...}. register is the "
    "Korean speech level a professional interpreter should use in this setting: one of \"합니다체\", \"해요체\", "
    "\"반말\". Change the situation only if the new sentences clearly show a different setting.")

EXTRACT_SYSTEM_V2 = (
    "You keep background notes for a live English-to-Korean interpreter of a talk. You get the current notes as "
    "JSON and the next sentences of the talk, each with the Korean translation already shown to the audience. "
    "Return JSON with exactly these keys and nothing else:\n"
    "\"summary\": the revised summary, at most 3 English sentences on what the talk is about and where it is now. "
    "Keep what is still true and change only what the new sentences change. Use null if the current summary "
    "needs no change.\n"
    "\"new_terms\": a list of {\"en\": ..., \"ko\": ...} for proper nouns that appear in the NEW sentences and "
    "are not in the current notes yet: names of people, places, organisations, companies, buildings and titles of "
    "works. Use the Korean rendering from the translations. Do not include common words or general terms. Empty "
    "list if there are none.\n"
    "\"situation\": {\"genre\": ..., \"speaker\": ..., \"audience\": ...} only if it is empty now or the new "
    "sentences clearly show a different setting, otherwise null. Do not decide the Korean speech level.\n"
    "Output only what changed: null for an unchanged summary or situation, an empty list for no new terms.")

REGISTER_V2 = ("\"situation\": {\"genre\": ..., \"speaker\": ..., \"audience\": ..., \"register\": ...} only if it is "
               "empty now or the new sentences clearly show a different setting, otherwise null. register is the Korean "
               "speech level a professional interpreter should use in this setting: one of \"합니다체\", \"해요체\", "
               "\"반말\".\n")


def extract_system_v2(with_register: bool) -> str:
    if not with_register:
        return EXTRACT_SYSTEM_V2
    head, tail = EXTRACT_SYSTEM_V2.split("\"situation\":", 1)
    return head + REGISTER_V2 + tail.split("\n", 1)[1]


EMPTY_CARD = {"summary": "", "terms": [], "situation": {}}


def extract_user(card: dict, rows: list[tuple[str, str]]) -> str:
    new = "\n".join(f"{k + 1}. EN: {en}\n   KO: {ko}" for k, (en, ko) in enumerate(rows))
    return f"Current notes:\n{json.dumps(card, ensure_ascii=False)}\n\nNew sentences:\n{new}"


def parse_card(raw: str, prev: dict) -> tuple[dict, bool]:
    """JSON 을 읽는다. 못 읽으면 이전 카드를 그대로 쓴다 (ok=False)."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("`").split("\n", 1)[-1]
    try:
        obj = json.loads(s[s.index("{"):s.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return prev, False
    card = {"summary": str(obj.get("summary", "")).strip(),
            "terms": [t for t in obj.get("terms", []) if isinstance(t, dict) and t.get("en") and t.get("ko")][:30],
            "situation": obj.get("situation") if isinstance(obj.get("situation"), dict) else {}}
    return card, True


def parse_card_v2(raw: str, prev: dict, new_src: list[str], at: int,
                  keep=("genre", "speaker", "audience")) -> tuple[dict, bool]:
    """V2 추출 결과를 합친다. 기존 항목은 그대로 두고(청중이 이미 봤다) 새 고유명사만 더한다. 항목마다 원문에
    마지막으로 나온 문장 번호(last_seen)를 적어 두고, MAX_TERMS 를 넘으면 가장 오래 안 나온 것부터 뺀다."""
    s = raw.strip()
    try:
        obj = json.loads(s[s.index("{"):s.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return prev, False
    terms = [dict(t) for t in prev["terms"]]
    text = " ".join(new_src).lower()
    for t in terms:
        if t["en"].lower() in text:
            t["last_seen"] = at
    have = {t["en"].lower() for t in terms}
    for t in obj.get("new_terms", []):
        if isinstance(t, dict) and t.get("en") and t.get("ko") and t["en"].lower() not in have:
            terms.append({"en": t["en"], "ko": t["ko"], "last_seen": at})
            have.add(t["en"].lower())
    terms = sorted(terms, key=lambda t: -t["last_seen"])[:MAX_TERMS]
    # 바뀐 칸만 온다. null·빈 값이면 이전 값을 그대로 쓴다.
    sit = obj.get("situation") if isinstance(obj.get("situation"), dict) and obj["situation"] else prev["situation"]
    sit = {k: v for k, v in sit.items() if k in keep}
    summary = obj.get("summary")
    summary = summary.strip() if isinstance(summary, str) and summary.strip() else prev["summary"]
    return {"summary": summary, "terms": terms, "situation": sit}, True


def render(cond: str, card: dict) -> str:
    parts = []
    for f in FIELDS[cond]:
        if f == "summary" and card["summary"]:
            parts.append("Summary of the talk so far: " + card["summary"])
        if f == "terms" and card["terms"]:
            parts.append("Names and terms already shown to the audience (English -> Korean):\n"
                         + "\n".join(f"- {t['en']} -> {t['ko']}" for t in card["terms"]))
        if f == "situation" and card["situation"]:
            s = card["situation"]
            parts.append("Situation: " + "; ".join(f"{k}: {s[k]}" for k in ("genre", "speaker", "audience", "register")
                                                     if s.get(k)))
    return "\n".join(parts)


def translate_user(current: str, notes: str) -> str:
    if not notes:
        return f"Translate the following English sentence into Korean.\n\n{current}"
    return f"<notes>\n{notes}\n</notes>\n\nTranslate ONLY the English sentence below into Korean.\n\n{current}"
