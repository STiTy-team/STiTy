"""Korean numbers: what the words mean, and parsing them into ints.

- `SINO` / `SMALL` / `BIG`: Sino-Korean digits and place values. `SINO_CHARS` is every
  character they use.
- `native_to_int`: native numerals as standalone words (하나 -> 1, 열 -> 10).
  `native_determiner_to_int`: the forms they take before a counter (한 병, 두 명 -> 1, 2).
- `sino_to_int`: 사백오십 -> 450, 이천삼백만 -> 23000000. None when the string is not a
  well-formed numeral (이이, 오늘).
- `AMOUNT` finds digits mixed with 억/만/천/백 (18만 7천 500, 5억 3천만) and `amount_to_int`
  turns one into an int (187500, 530000000).
"""

import re

SINO = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
SMALL = {"십": 10, "백": 100, "천": 1000}
BIG = {"만": 10**4, "억": 10**8}
SINO_CHARS = "".join([*SINO, *SMALL, *BIG])
_NATIVE = {
    "하나": 1, "둘": 2, "셋": 3, "넷": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9,
    "열": 10,
}  # fmt: skip
_NATIVE_DETERMINERS = {
    "한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9,
    "열": 10,
}  # fmt: skip
AMOUNT = re.compile(r"\d+\s*[억만천백](?:[만억]|\s*\d+\s*[억만천백])*(?:\s*\d+)?\s*")


def native_to_int(word: str) -> int | None:
    return _NATIVE.get(word)


def native_determiner_to_int(word: str) -> int | None:
    return _NATIVE_DETERMINERS.get(word)


def sino_to_int(numeral: str) -> int | None:
    total, section, current = 0, 0, 0
    previous_digit = False
    for ch in numeral:
        if ch in SINO:
            if previous_digit:
                return None
            current, previous_digit = SINO[ch], True
            continue
        previous_digit = False
        if ch in SMALL:
            section += (current or 1) * SMALL[ch]
            current = 0
        elif ch in BIG:
            total += (section + current or 1) * BIG[ch]
            section, current = 0, 0
        else:
            return None
    return total + section + current


def amount_to_int(amount: str) -> int:
    total, section = 0, 0
    for number, unit in re.findall(r"(\d*)\s*([억만천백])", amount):
        value = int(number or 0)
        if unit in BIG:
            total += (section + value) * BIG[unit]
            section = 0
        else:
            section += value * SMALL[unit]
    tail = re.search(r"[억만천백]\s*(\d+)$", amount.strip())
    return total + section + (int(tail.group(1)) if tail else 0)
