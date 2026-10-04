"""Korean sentence-ending register of hypotheses in a jsonl (field hyp): split at <SEG> / sentence
punctuation, classify each piece's ending as banmal / hapnida / haeyo / other."""
import json, re, sys
from collections import Counter
def cls(piece):
    p = re.sub(r"[\s.!?…\"'”’)]+$", "", piece)
    if not p: return None
    if re.search(r"(습니다|ㅂ니다|입니다|됩니다|십시오|니까)$", p): return "hapnida"
    if re.search(r"요$", p): return "haeyo"
    if re.search(r"(다|야|어|아|지|니|냐|라|자|네|군|데|해|워|봐|줘|게)$", p): return "banmal/plain"
    return "other"
for path in sys.argv[1:]:
    c = Counter(); n = 0
    for l in open(path, encoding="utf-8"):
        h = json.loads(l)["hyp"]
        for piece in re.split(r"<SEG>|(?<=[.!?])\s+", h):
            k = cls(piece)
            if k: c[k] += 1; n += 1
    tot = max(1, n)
    print(f"{path.split('/')[-1]}: " + "  ".join(f"{k} {v/tot:.2f}" for k, v in sorted(c.items())) + f"  (n={n})")
