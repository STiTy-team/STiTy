"""Check that r2norm aligns what it should and leaves real errors alone. From the repo root: python -m bench.metrics.test_r2norm"""
import os
import sys

os.environ["STITY_R2NORM"] = "all"
from bench.metrics import transcription  # noqa: E402

# (reference, transcript, should they match after normalization)
CASES = [
    ("Mostly. There's still the, um… the launch-date thing.", "Mostly there's still the umm the launch date thing", True),
    ("Then the ball's in our court for the, uh— the Bluetooth fix.", "then the ball's in our court for the the bluetooth fix", True),
    ("어, 어어 과장님, 소맥 제가 말까요?", "과장님 소맥 제가 말까요", True),
    ("흠.. 목살 좀 더 시켜야 되나…", "음 목살 쫌 더 시켜야 되나", True),
    ("다 잡힌 것 같아요", "다 잡힌 거 같애요", True),
    ("무슨 소리예요", "무슨 소리에요", True),
    ("걸리고요", "걸리구요", True),
    ("Okay, okay", "OK ok", True),
    ("물냉, 비냉 중에 뭐로 하실래요?", "물냉 비냉 중에 뭘로 하실래요", True),
    ("크래시", "크래쉬", True),
    ("J-M-T? What's JMT?", "존맛탱? What's 존맛탱?", True),
    ("진짜 JMT이예요.", "진짜 존맛탱이에요", True),
    ("부산 가시믄 해운대 가가 회 접시 한 접시 드이소. 광안리 야경도 직입니더.",
     "부산 가시면 해운대 가서 회 접시 한 접시 드세요 광안리 야경도 직입니다", True),
    ("넘어갔제! 롯데 올해는 진짜 되는갑다!", "넘어갔지 롯데 올해는 진짜 되는가 보다", True),
    ("하하하하, 2차요?", "하하하하 이차요", True),
    ("아 2차 2차 2차 없습니다!", "아 이차 이차 이차 없습니다", True),
    ("소주 한 잔, 맥주 두 병", "소주 한잔 맥주 두병", True),
    # differences that must remain
    ("아, 네. 그거 제가", "네 그거 제가", False),
    ("Oh, crap — sorry, sorry! Did that— ah, did it get on your phone?", "oh crap sorry sorry did that did it get on your phone", False),
    ("광안리 야경도 직입니더.", "광안리 야경도 찍입니다", False),
    ("팔로업 메일", "팔로우 메일", False),
    ("J-M-T? What's JMT?", "John Martin? What's John Martin?", False),
    ("that's won, right?", "that's one right", False),
    ("응응, 응. 나 아홉 시 반까진 갈게.", "나 9시 반까진 갈게", False),
]
bad = 0
for ref, hyp, same in CASES:
    row = {"src_lang": "ko", "reference": ref, "transcription_output": hyp}
    (r, h), = transcription._transcripts([row])
    ok = ("".join(r.split()) == "".join(h.split())) == same
    bad += not ok
    print("ok " if ok else "BAD", "|", r, "||", h)
print("all passed" if not bad else f"{bad} failed")
sys.exit(1 if bad else 0)
