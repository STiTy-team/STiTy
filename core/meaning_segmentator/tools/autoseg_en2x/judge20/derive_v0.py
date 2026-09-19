"""v0-A 에서 B·C 를 **기계적으로** 파생한다 — 통제 비교를 위해서다.

A·B·C 가 [Core Principles] 까지 한 글자도 다르지 않아야 차이가 전부 `[Scoring Rules]` 몫이
된다. 골격을 바꿔 Writer 에게 다시 생성시키면 v0 생성 편차(런 사이 0.021, 잡음과 구분 안 됨)가
섹션 효과 위에 얹혀 0.007 짜리를 못 본다.

    A  그대로
    B  [Scoring Rules] 통째 삭제
    C  그 섹션에 "모순은 등급화된 확률이지 거부권이 아니다" 한 줄만 남김

B·C 는 `[Output Rules]` 의 상호참조도 함께 고친다 — 없어진(또는 목적함수를 더는 정의하지
않는) 섹션을 가리키게 두면 지시가 허공을 가리킨다. 그래서 **B 와 C 의 차이는 정확히 그 한
줄**이다.

근거: probe_port 이식 실험에서 채점 척도 설명(등급표)을 뺀 p3 가 0.5738 로 가장 높았고
(v0 0.5669), 구체적 표면 단서를 뺀 p2 는 0.5656 으로 내려갔다. 프롬프트가 값을 하는 쪽은
무엇이 좋은 절단인지 설명하는 부분이 아니라 어디를 보라고 지목하는 부분이다.
"""
import sys
from pathlib import Path

sys.path.insert(0, '.')
from core.meaning_segmentator.autoseg.runtime.agents_judge import JUDGE_SECTIONS

A = Path('core/meaning_segmentator/experiment/artifacts/en2x/en-multi/judge20')
OUT = A / 'prompts'
SEC = '[Scoring Rules]'
XREF = ("how good a cut at that position is for streaming translation, exactly as the "
        "measured target stated in the Scoring Rules section defines it.")
PLAIN = "how good a cut at that position is for streaming translation."
KEEP_LINE = (
    "- Contradiction risk introduced by the continuation is a GRADED probability that discounts "
    "a position continuously. Do NOT convert it into a binary flag or a separate tier.")


def drop_section(prompt: str, header: str) -> str:
    i = prompt.find(header)
    assert i >= 0, f'섹션 없음: {header}'
    nxt = [prompt.find(s, i + len(header)) for s in JUDGE_SECTIONS if s != header]
    j = min([x for x in nxt if x >= 0], default=len(prompt))
    return (prompt[:i] + prompt[j:]).replace('\n\n\n', '\n\n')


def replace_section(prompt: str, header: str, body: str) -> str:
    i = prompt.find(header)
    assert i >= 0, f'섹션 없음: {header}'
    nxt = [prompt.find(s, i + len(header)) for s in JUDGE_SECTIONS if s != header]
    j = min([x for x in nxt if x >= 0], default=len(prompt))
    return prompt[:i] + f'{header}\n{body}\n\n' + prompt[j:]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    a = (A / 'prompt_v0.txt').read_text(encoding='utf-8')
    assert XREF in a, 'Output Rules 의 상호참조 문면이 예상과 다르다 — 확인 후 고칠 것'

    b = drop_section(a, SEC).replace(XREF, PLAIN)
    c = replace_section(a, SEC, KEEP_LINE).replace(XREF, PLAIN)

    assert SEC not in b, 'B 에 섹션이 남았다'
    assert SEC in c and 'cohesion x (1 - contra)' not in c, 'C 가 목적함수를 아직 말한다'
    for name, pr in (('A', a), ('B', b), ('C', c)):
        (OUT / f'v0_{name}.txt').write_text(pr, encoding='utf-8')
        print(f'[{name}] {len(pr)}자 / 섹션 '
              + ' '.join(s for s in JUDGE_SECTIONS if s in pr))
    # 통제 확인 — [Core Principles] 가 셋에서 동일해야 한다
    def principles(p):
        i = p.find('[Core Principles]')
        nxt = [p.find(s, i + 1) for s in JUDGE_SECTIONS if s != '[Core Principles]']
        return p[i:min(x for x in nxt if x >= 0)]
    assert principles(a) == principles(b) == principles(c), '[Core Principles] 가 다르다'
    print('[통제] [Core Principles] 세 갈래 동일 확인')
    print(f'[차이] B vs C = 한 줄({len(KEEP_LINE)}자): 모순은 등급화된 확률이지 거부권이 아니다')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
