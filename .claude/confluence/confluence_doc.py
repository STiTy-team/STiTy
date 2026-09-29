#!/usr/bin/env python3
"""업무 공유 문서를 Confluence STiTy 스페이스에 만든다.

가이드 문서(업무 공유 작성 가이드)의 구조를 그대로 재현한다.
스킬(.claude/commands/confluence-report.md)이 JSON 을 만들어 이 스크립트에 넘기는
방식이다. 인증은 .env 의 ATLASSIAN_EMAIL/ATLASSIAN_API_TOKEN.

  python .claude/confluence/confluence_doc.py --show-format
  python .claude/confluence/confluence_doc.py --list-folders 5
  python .claude/confluence/confluence_doc.py --json payload.json --dry-run
  python .claude/confluence/confluence_doc.py --json payload.json
"""
import argparse
import html
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]

# storage 형식은 ac:/ri: 접두사를 그대로 요구한다. 등록하지 않으면 ns0:/ns1: 로 나가서
# Confluence 가 매크로와 페이지 링크를 알아보지 못한다.
ET.register_namespace("ac", "http://atlassian.com/content")
ET.register_namespace("ri", "http://atlassian.com/resource/identifier")
CONFIG = json.loads((Path(__file__).parent / "config.json").read_text(encoding="utf-8"))

KIND_KO = "공유"


def load_env() -> tuple[str, str]:
    """.env 에서 이메일과 토큰을 읽는다. 환경변수가 이미 있으면 그쪽을 쓴다."""
    env = {}
    envfile = ROOT / ".env"
    if envfile.exists():
        for line in envfile.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    email = os.environ.get("ATLASSIAN_EMAIL") or env.get("ATLASSIAN_EMAIL")
    token = os.environ.get("ATLASSIAN_API_TOKEN") or env.get("ATLASSIAN_API_TOKEN")
    if not email or not token:
        sys.exit("ATLASSIAN_EMAIL / ATLASSIAN_API_TOKEN 이 .env 에 있어야 한다.")
    return email, token


class Confluence:
    def __init__(self, email: str, token: str):
        self.base = CONFIG["base_url"]
        self.client = httpx.Client(auth=(email, token), timeout=30.0,
                                   headers={"Accept": "application/json"})

    def _check(self, r: httpx.Response) -> dict:
        if r.status_code >= 400:
            sys.exit(f"Confluence API 실패 {r.status_code}: {r.text[:600]}")
        return r.json() if r.content else {}

    def folders_under(self, parent_id: str) -> list[tuple[str, str]]:
        """parent_id 바로 아래 폴더들의 (제목, id)."""
        r = self.client.get(f"{self.base}/rest/api/search", params={
            "cql": f'space={CONFIG["space_key"]} and type=folder and parent={parent_id}',
            "limit": 100,
        })
        return [(it.get("title", ""), it["content"]["id"])
                for it in self._check(r).get("results", [])]

    def round_folder(self, n: str) -> tuple[str, str] | None:
        """차수 폴더를 찾는다.

        폴더 이름이 차수마다 다르다 — '1차 업무 분담', '3,4차 업무 분담', '5차 업무 공유'.
        그래서 이름 앞의 차수 숫자만 보고 고른다. '3,4차' 는 3 과 4 둘 다에 걸린다.
        """
        for title, fid in self.folders_under(CONFIG["docs_folder_id"]):
            m = re.match(r"^\s*([\d,\s]+)차", title)
            if m and n in [x.strip() for x in m.group(1).split(",")]:
                return title, fid
        return None

    def folder_tree(self, parent_id: str, depth: int = 0) -> list[tuple[int, str, str]]:
        """(깊이, 제목, id) 를 위에서부터 차례로."""
        out = []
        for title, fid in sorted(self.folders_under(parent_id)):
            out.append((depth, title, fid))
            out += self.folder_tree(fid, depth + 1)
        return out

    def create_folder(self, parent_id: str, title: str) -> str:
        r = self.client.post(f"{self.base}/api/v2/folders", json={
            "spaceId": CONFIG["space_id"], "title": title, "parentId": parent_id,
        })
        return self._check(r)["id"]

    def get_storage(self, page_id: str) -> str:
        r = self.client.get(f"{self.base}/rest/api/content/{page_id}",
                            params={"expand": "body.storage"})
        return self._check(r)["body"]["storage"]["value"]

    def find_page(self, title: str) -> str | None:
        r = self.client.get(f"{self.base}/rest/api/content",
                            params={"spaceKey": CONFIG["space_key"], "title": title, "limit": 1})
        results = self._check(r).get("results", [])
        return results[0]["id"] if results else None

    def create_page(self, parent_id: str, title: str, body: str) -> dict:
        r = self.client.post(f"{self.base}/api/v2/pages", json={
            "spaceId": CONFIG["space_id"], "status": "current", "title": title,
            "parentId": parent_id,
            "body": {"representation": "storage", "value": body},
        })
        return self._check(r)

    def update_page(self, page_id: str, title: str, body: str) -> dict:
        """기존 페이지의 본문을 갈아끼운다.

        Confluence 는 버전 번호를 낙관적 잠금으로 쓴다. 지금 버전을 읽어 +1 해서
        보내야 하고, 그 사이 남이 고쳤으면 409 로 막힌다 — 남의 수정을 조용히
        덮어쓰지 않게 하는 장치이므로 강제로 뚫지 않는다.
        """
        r = self.client.get(f"{self.base}/api/v2/pages/{page_id}",
                            params={"body-format": "storage"})
        current = self._check(r)
        r = self.client.put(f"{self.base}/api/v2/pages/{page_id}", json={
            "id": page_id, "status": "current", "title": title,
            "body": {"representation": "storage", "value": body},
            "version": {"number": current["version"]["number"] + 1,
                        "message": "confluence_doc.py --update"},
        })
        return self._check(r)

    def set_owner(self, page_id: str, account_id: str) -> dict:
        """페이지 소유자를 바꾼다. 스크립트는 관리 계정(스티티)으로 올리므로 그대로 두면
        소유자가 전부 스티티가 된다 — 작성자로 적힌 사람을 소유자로 돌린다.

        v2 는 소유자만 따로 바꾸는 끝점이 없어서, 본문·제목을 그대로 둔 채 버전을 +1 하고
        `ownerId` 를 실어 PUT 한다. 판 하나가 더 생기지만 내용은 같다.
        """
        r = self.client.get(f"{self.base}/api/v2/pages/{page_id}",
                            params={"body-format": "storage"})
        cur = self._check(r)
        if cur.get("ownerId") == account_id:
            return cur
        r = self.client.put(f"{self.base}/api/v2/pages/{page_id}", json={
            "id": page_id, "status": "current", "title": cur["title"],
            "body": {"representation": "storage", "value": cur["body"]["storage"]["value"]},
            "version": {"number": cur["version"]["number"] + 1,
                        "message": "confluence_doc.py: 작성자를 소유자로"},
            "ownerId": account_id,
        })
        return self._check(r)

    def find_user(self, name: str) -> str | None:
        """표시 이름이 정확히 같은 Atlassian 계정 하나의 accountId. 없거나 여럿이면 None."""
        r = self.client.get(f"{CONFIG['base_url'].rsplit('/wiki', 1)[0]}/rest/api/3/user/search",
                            params={"query": name, "maxResults": 20})
        hits = [u for u in self._check(r)
                if u.get("accountType") == "atlassian" and u.get("displayName") == name]
        return hits[0]["accountId"] if len(hits) == 1 else None

    def add_labels(self, page_id: str, labels: list[str]) -> None:
        if not labels:
            return
        r = self.client.post(f"{self.base}/rest/api/content/{page_id}/label",
                             json=[{"prefix": "global", "name": n} for n in labels])
        self._check(r)


def esc(s: str) -> str:
    return html.escape(str(s or ""), quote=True)


# 문서 안 링크. 결론의 각 항목이 업무 내용 정리의 해당 소제목으로 바로 내려가게 한다.
#
# 소제목 번호가 곧 앵커다 — `# 1. 배경` 은 sec-1, `## 1.2 한 일` 은 sec-1-2. 결론에서
# `[텍스트](#1.2)` 로 걸면 그 소제목으로 간다. Confluence 가 소제목에 자동으로 다는
# 앵커는 제목 문구에서 만들어져 문구가 바뀌면 깨지므로, anchor 매크로를 직접 박는다.
# 렌더링하는 동안 생긴 앵커와 걸린 링크를 모아 두었다가, 짝이 없는 링크가 있으면 멈춘다.
_ANCHORS: set[str] = set()
_LINKS: list[str] = []
SEC_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s")


def anchor_id(num: str) -> str:
    return "sec-" + num.strip(".").replace(".", "-")


def anchor_macro(aid: str) -> str:
    return ('<ac:structured-macro ac:name="anchor" ac:schema-version="1">'
            f'<ac:parameter ac:name="">{esc(aid)}</ac:parameter></ac:structured-macro>')


def link(text: str, target: str) -> str:
    if target.startswith("#"):
        aid = anchor_id(target[1:])
        _LINKS.append(aid)
        return (f'<ac:link ac:anchor="{esc(aid)}">'
                f"<ac:link-body>{esc(text)}</ac:link-body></ac:link>")
    return f'<a href="{esc(target)}">{esc(text)}</a>'


INLINE_RE = re.compile(r"`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^)\s]+\)")
LINK_RE = re.compile(r"^\[([^\]]+)\]\(([^)\s]+)\)$")
LIST_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def inline(text: str) -> str:
    """줄 안의 `코드`, **굵게**, [링크](대상) 만 살린다. 나머지는 그대로 escape."""
    out, pos = [], 0
    for m in INLINE_RE.finditer(text):
        out.append(esc(text[pos:m.start()]))
        tok = m.group(0)
        if tok.startswith("`"):
            out.append(f"<code>{esc(tok[1:-1])}</code>")
        elif tok.startswith("["):
            lm = LINK_RE.match(tok)
            out.append(link(lm.group(1), lm.group(2)))
        else:
            out.append(f"<strong>{esc(tok[2:-2])}</strong>")
        pos = m.end()
    out.append(esc(text[pos:]))
    return "".join(out)


# 코드 블록 본문을 잠시 담아 두는 자리. 아래 두 가지가 겹쳐서 이렇게 돌아간다.
#
#  1. Confluence 는 CDATA 없는 `ac:plain-text-body` 를 통째로 버린다. 이스케이프만
#     해서 넣으면 빈 매크로(`<ac:structured-macro ... />`)로 저장돼 문서에 빈 칸만
#     남는다.
#  2. 그런데 render() 가 조립한 XHTML 을 ElementTree 로 파싱·재직렬화하는데, ET 는
#     CDATA 를 보존하지 않고 일반 텍스트로 풀어 버린다. 그래서 code_macro 에서
#     CDATA 를 넣어 봐야 최종 산출물에는 안 남는다.
#
# 그래서 직렬화까지는 표식만 들고 가고, 다 끝난 문자열에서 CDATA 로 바꾼다.
_CODE_BODIES: list[str] = []
CODE_MARK_RE = re.compile(r"@@CODEBLOCK(\d+)@@")


def code_macro(text: str, lang: str = "") -> str:
    param = f'<ac:parameter ac:name="language">{esc(lang)}</ac:parameter>' if lang else ""
    _CODE_BODIES.append(str(text or ""))
    # 표식은 XML 에 넣을 수 있는 문자여야 한다. NUL 을 쓰면 ElementTree 가
    # 파싱 단계에서 버린다.
    mark = f"@@CODEBLOCK{len(_CODE_BODIES) - 1}@@"
    return ('<ac:structured-macro ac:name="code" ac:schema-version="1">' + param +
            f"<ac:plain-text-body>{mark}</ac:plain-text-body>"
            "</ac:structured-macro>")


def restore_code_bodies(body: str) -> str:
    """표식을 실제 코드 본문(CDATA)으로 되돌린다. 직렬화가 끝난 뒤에 부른다."""
    def sub(m):
        raw = _CODE_BODIES[int(m.group(1))]
        return "<![CDATA[" + raw.replace("]]>", "]]]]><![CDATA[>") + "]]>"
    return CODE_MARK_RE.sub(sub, body)


def build_list(items: list[tuple[int, bool, list[str]]], idx: int, depth: int) -> tuple[str, int]:
    """(들여쓰기 깊이, 번호목록 여부, 문단들) 목록을 <ul>/<ol> 로 접는다. 중첩도 살린다.

    한 항목에 문단이 여럿이면 항목 안에서 줄을 바꿔 잇는다 — 결론의 `1. 링크` 아래
    들여 쓴 설명 줄이 이 꼴이다.
    """
    ordered = items[idx][1]
    tag = "ol" if ordered else "ul"
    parts = ['<ol start="1">' if ordered else "<ul>"]
    while idx < len(items):
        d, o, paras = items[idx]
        if d < depth or (d == depth and o != ordered):
            break
        if d > depth:
            sub, idx = build_list(items, idx, d)
            parts[-1] = parts[-1][: -len("</li>")] + sub + "</li>"
            continue
        parts.append("<li>" + "".join(f"<p>{inline(t)}</p>" for t in paras) + "</li>")
        idx += 1
    parts.append(f"</{tag}>")
    return "".join(parts), idx


TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def split_row(line: str) -> list[str]:
    """`| a | b |` 를 칸 목록으로 쪼갠다. 양끝 파이프는 있어도 없어도 된다."""
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [c.strip() for c in body.split("|")]


def is_table_start(lines: list[str], i: int) -> bool:
    """머리글 줄 다음에 구분선(`|---|---|`)이 와야 표로 본다."""
    return (lines[i].strip().startswith("|")
            and i + 1 < len(lines) and TABLE_SEP_RE.match(lines[i + 1]) is not None)


def take_table(lines: list[str], i: int) -> tuple[str, int]:
    header = split_row(lines[i])
    i += 2                                   # 머리글 + 구분선
    rows = []
    while i < len(lines) and lines[i].strip().startswith("|"):
        rows.append(split_row(lines[i]))
        i += 1
    ncol = len(header)

    def pad(cells):
        return (cells + [""] * ncol)[:ncol]   # 칸 수가 어긋나도 표가 깨지지 않게 맞춘다

    parts = ['<table data-layout="default"><tbody><tr>']
    parts += [f"<th><p>{inline(c)}</p></th>" for c in pad(header)]
    parts.append("</tr>")
    for r in rows:
        parts.append("<tr>" + "".join(
            f"<td><p>{inline(c)}</p></td>" for c in pad(r)) + "</tr>")
    parts.append("</tbody></table>")
    return "".join(parts), i


def take_list(lines: list[str], i: int) -> tuple[str, int]:
    items = []
    while i < len(lines):
        m = LIST_RE.match(lines[i])
        if m:
            items.append((len(m.group(1)) // 2, m.group(2)[0] not in "-*+", [m.group(3).strip()]))
            i += 1
            continue
        # 목록 표식 없이 들여 쓴 줄은 바로 위 항목 안의 다음 문단이다.
        if items and lines[i].strip() and lines[i][:1] in " 	":
            items[-1][2].append(lines[i].strip())
            i += 1
            continue
        break
    return build_list(items, 0, items[0][0])[0], i


def markdown(text: str, base: int) -> str:
    """md 로 쓴 내용을 storage 형식으로 바꾼다.

    소제목(#), 목록(- / 1.), 표(| a | b |), 코드블록(```), 문단, 인라인 `코드`·**굵게**·
    [링크](대상) 를 지원한다. 링크 대상이 `#1.2` 면 번호가 1.2 인 소제목으로 가는 문서 안
    링크, 그 밖에는 일반 URL 이다.

    `#` 은 h(base+1) 로 그린다 — 가이드의 절 제목 바로 아래 단계.
    번호로 시작하는 소제목(`1.`, `1.2`)에는 그 번호로 앵커를 단다.
    """
    lines = str(text or "").replace("\r\n", "\n").split("\n")
    out: list[str] = []
    para: list[str] = []

    def flush():
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()

    i = 0
    while i < len(lines):
        line, stripped = lines[i], lines[i].strip()
        if not stripped:
            flush()
            i += 1
            continue
        m = HEADING_RE.match(stripped)
        if m:
            flush()
            lv = min(len(m.group(1)) + base, 6)
            num = SEC_NUM_RE.match(m.group(2))
            mark = ""
            if num:
                aid = anchor_id(num.group(1))
                _ANCHORS.add(aid)
                mark = anchor_macro(aid)
            out.append(f"<h{lv}>{mark}{inline(m.group(2))}</h{lv}>")
            i += 1
            continue
        if stripped.startswith("```"):
            flush()
            lang = stripped[3:].strip()
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            out.append(code_macro("\n".join(buf), lang))
            i += 1
            continue
        if is_table_start(lines, i):
            flush()
            block, i = take_table(lines, i)
            out.append(block)
            continue
        if LIST_RE.match(line):
            flush()
            block, i = take_list(lines, i)
            out.append(block)
            continue
        para.append(stripped)
        i += 1
    flush()
    return "".join(out) or "<p />"


def cell(value, kind: str, base: int) -> str:
    """JSON 의 값 하나를 storage 형식으로 바꾼다."""
    if kind == "markdown":
        return markdown(value, base)
    return f"<p>{inline(str(value or ''))}</p>"


def parse_storage(xhtml: str) -> ET.Element:
    """storage 형식을 파싱한다. ac:/ri: 접두사와 &nbsp; 때문에 손질이 필요하다."""
    text = re.sub(r"&(?!(amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)([a-zA-Z]+);",
                  lambda m: html.unescape(f"&{m.group(2)};"), xhtml)
    wrapper = ('<root xmlns:ac="http://atlassian.com/content" '
               'xmlns:ri="http://atlassian.com/resource/identifier">'
               f"{text}</root>")
    return ET.fromstring(wrapper)


HEADINGS = ("h1", "h2", "h3", "h4", "h5", "h6")


def text_of(el: ET.Element) -> str:
    return "".join(el.itertext()).strip()


def row_labels(root: ET.Element) -> list[str]:
    """표의 왼쪽 항목 이름들을 순서대로 뽑는다."""
    names = []
    for tr in root.iter("tr"):
        th = tr.find("th")
        if th is not None and text_of(th):
            names.append(text_of(th))
    return names


def prose_labels(root: ET.Element) -> list[str]:
    """표 밖의 항목 — 가이드 본문 맨 윗단의 소제목 하나가 항목 하나다.
    `한 줄 요약`·`결론`·`업무 내용 정리` 처럼 칸에 가두지 않고 줄글로 쓰는 항목이다."""
    return [text_of(el) for el in root if el.tag in HEADINGS and text_of(el)]


def field_labels(root: ET.Element) -> list[str]:
    return row_labels(root) + prose_labels(root)


def render(template: str, fields: dict) -> tuple[str, list[str]]:
    """가이드 본문을 틀로 삼아 값을 끼워 넣는다.

    표는 왼쪽 항목 이름으로, 줄글은 소제목 이름으로 짝을 맞춘다. 값이 없는 항목은
    가이드의 안내 문구를 그대로 두면 안 되므로 비운다.
    반환값의 두 번째는 JSON 에 값이 없어서 비워둔 항목 목록이다.
    """
    root = parse_storage(template)
    missing = []

    for tr in root.iter("tr"):
        th, td = tr.find("th"), tr.find("td")
        if th is None or td is None or not text_of(th):
            continue
        spec = fields.get(text_of(th))
        if spec is None:
            missing.append(text_of(th))
            filled = "<p />"
        else:
            filled = cell(spec.get("value"), spec.get("type", "text"), base=3)
        for child in list(td):
            td.remove(child)
        td.text = None
        for node in parse_storage(filled):
            td.append(node)

    # 가이드 맨 위, 첫 표나 소제목 앞에 오는 문단은 제목 작성 안내다. 실제 문서에는 뺀다.
    for el in list(root):
        if el.tag in HEADINGS or el.tag == "table":
            break
        root.remove(el)

    # 줄글 항목: 소제목 다음부터 다음 소제목 전까지가 그 항목의 자리다. 가이드의
    # 안내 문단을 걷어내고 값을 끼운다. 값 안의 소제목은 항목 소제목보다 한 단계 아래다.
    sections = [el for el in root if el.tag in HEADINGS and text_of(el)]
    for el in sections:
        children = list(root)
        j = children.index(el) + 1
        while j < len(children) and children[j].tag not in HEADINGS:
            root.remove(children[j])
            j += 1
        name = text_of(el)
        spec = fields.get(name)
        if spec is None:
            missing.append(name)
            filled = "<p />"
        else:
            filled = cell(spec.get("value"), spec.get("type", "text"), base=int(el.tag[1]))
        pos = list(root).index(el) + 1
        for node in parse_storage(filled):
            root.insert(pos, node)
            pos += 1

    body = "".join(ET.tostring(child, encoding="unicode") for child in root)
    body = re.sub(r"\sxmlns:(ac|ri)=\"[^\"]*\"", "", body)
    return restore_code_bodies(body), missing


MAX_SEQ = 50


def build_title(d: dict, seq: int | None) -> str:
    """제목: [n차][세분화 업무명] 공유 문서. 같은 제목이 있으면 [n차][업무명][2] 공유 문서."""
    tail = f"[{seq}]" if seq else ""
    return f'[{d["round"]}차][{d["task"]}]{tail} {KIND_KO} 문서'


def resolve_title(cf: Confluence, d: dict, autonumber: bool) -> str:
    """쓸 수 있는 제목을 고른다.

    JSON 에 seq 를 주면 그 번호를 그대로 쓴다. 안 주면 같은 제목이 이미 있는지 보고
    비어 있는 다음 번호를 찾는다.
    """
    seq = d.get("seq")
    if seq:
        return build_title(d, int(seq))

    title = build_title(d, None)
    if not autonumber or not cf.find_page(title):
        return title

    for n in range(2, MAX_SEQ + 1):
        candidate = build_title(d, n)
        if not cf.find_page(candidate):
            return candidate
    sys.exit(f"같은 제목의 문서가 {MAX_SEQ} 개를 넘었다. seq 로 직접 번호를 지정할 것.")


def slug(s: str) -> str:
    """Confluence 라벨은 공백을 못 쓴다."""
    return str(s).replace(" ", "")


def resolve_folder(cf: Confluence, d: dict) -> tuple[str, str, bool]:
    """문서를 넣을 폴더의 (id, 경로 표시, 새로 만들어야 하는지).

    folder_id 는 --list-folders 로 본 차수 폴더 트리 안에 있어야 한다. 다른 차수의
    폴더 id 를 잘못 옮겨 적으면 문서가 엉뚱한 차수에 들어가므로 여기서 막는다.
    new_folder 를 주면 folder_id 아래에 그 이름으로 폴더를 만들어 넣는다.
    """
    found = cf.round_folder(str(d["round"]))
    if not found:
        sys.exit(f'{d["round"]}차 폴더가 없다. --list-folders {d["round"]} 로 확인할 것.')
    round_title, round_id = found
    fid = str(d.get("folder_id") or "")
    if not fid:
        sys.exit("folder_id 가 필요하다. --list-folders 로 고른 폴더 id 를 넣을 것.")

    paths = {round_id: round_title}
    stack = [round_title]
    for depth, title, i in cf.folder_tree(round_id):
        stack = stack[:depth + 1] + [title]
        paths[i] = " > ".join(stack)
    if fid not in paths:
        sys.exit(f"folder_id {fid} 가 '{round_title}' 아래에 없다.")

    new = (d.get("new_folder") or "").strip()
    if new:
        existing = dict(cf.folders_under(fid)).get(new)
        if existing:
            return existing, f"{paths[fid]} > {new}", False
        return fid, f"{paths[fid]} > {new}", True
    return fid, paths[fid], False


def check_links() -> None:
    """결론의 [..](#1.2) 가 가리키는 소제목이 업무 내용 정리에 실제로 있는지 본다."""
    broken = sorted({a for a in _LINKS if a not in _ANCHORS})
    if broken:
        have = ", ".join(sorted(_ANCHORS)) or "(없음)"
        sys.exit(f"짝이 없는 문서 안 링크: {', '.join(broken)}\n"
                 f"번호가 붙은 소제목으로 생긴 앵커: {have}")


def preview(title: str, folder: str, labels: list[str], fields: dict) -> str:
    """사람이 읽고 검토할 전문. storage XHTML 대신 JSON 에 쓴 글을 그대로 보인다."""
    out = [f"제목  : {title}", f"폴더  : {folder}", f"라벨  : {', '.join(labels)}", ""]
    for name, spec in fields.items():
        value = str(spec.get("value") or "")
        if spec.get("type") == "markdown" or "\n" in value:
            out += [f"=== {name} ===", value, ""]
        else:
            out.append(f"{name}: {value}")
    return "\n".join(out)


def assign_owner(cf: Confluence, page_id: str, fields: dict) -> None:
    """`작성자` 칸의 이름이 Atlassian 계정과 정확히 맞으면 그 사람을 소유자로 만든다.

    이름이 없거나 둘 이상 맞으면 건드리지 않고 알려만 준다 — 엉뚱한 사람을 소유자로
    만드는 것보다 관리 계정으로 남는 쪽이 낫다.
    """
    spec = fields.get("작성자") or {}
    name = str(spec.get("value") or "").strip()
    if not name:
        return
    account = cf.find_user(name)
    if not account:
        print(f"소유자: '{name}' 과 정확히 같은 계정이 하나가 아니라 바꾸지 않았다")
        return
    try:
        cf.set_owner(page_id, account)
        print(f"소유자: {name}")
    except SystemExit as e:
        print(f"소유자 변경 실패: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="업무 공유 문서를 Confluence 에 만든다. 형식은 가이드 문서에서 그때그때 읽는다.")
    ap.add_argument("--json", help="문서 내용을 담은 JSON 파일")
    ap.add_argument("--show-format", action="store_true",
                    help="가이드에서 현재 형식(항목 이름)만 읽어 출력. JSON 을 짜기 전에 먼저 볼 것")
    ap.add_argument("--list-folders", metavar="N",
                    help="N차 폴더 아래의 하위 폴더 트리를 id 와 함께 출력")
    ap.add_argument("--dry-run", action="store_true",
                    help="올리지 않고 검토용 전문과 본문 크기만 출력")
    ap.add_argument("--show-body", action="store_true",
                    help="--dry-run 에서 storage XHTML 본문까지 출력")
    ap.add_argument("--update", action="store_true",
                    help="같은 제목의 문서가 이미 있으면 번호를 붙이지 않고 본문을 갱신한다")
    ap.add_argument("--no-autonumber", action="store_true",
                    help="같은 제목이 있어도 번호를 붙이지 않고 그냥 멈춘다")
    args = ap.parse_args()

    cf = Confluence(*load_env())

    if args.show_format:
        guide_id = CONFIG["guide"]
        root = parse_storage(cf.get_storage(guide_id))
        print(f"가이드 페이지: {guide_id}")
        print("채워야 하는 항목 (이 이름을 JSON 의 fields 키로 그대로 쓸 것):")
        for name in field_labels(root):
            print(f"  - {name}")
        return

    if args.list_folders:
        found = cf.round_folder(args.list_folders.strip())
        if not found:
            print(f"{args.list_folders}차 폴더가 없다. 차수 폴더 목록 (부모 {CONFIG['docs_folder_id']}):")
            for title, fid in sorted(cf.folders_under(CONFIG["docs_folder_id"])):
                print(f"  {title}  ({fid})")
            return
        title, fid = found
        print(f"{title}  ({fid})")
        for depth, t, i in cf.folder_tree(fid):
            print(f"{'  ' * (depth + 1)}{t}  ({i})")
        return

    if not args.json:
        sys.exit("--json, --show-format, --list-folders 중 하나가 필요하다.")

    d = json.loads(Path(args.json).read_text(encoding="utf-8"))
    for key in ("round", "task"):
        if not d.get(key):
            sys.exit(f"필수 항목 누락: {key}  (제목 형식: [n차][세분화 업무명] 공유 문서)")

    # --update 는 있는 문서를 고치는 것이므로 번호를 붙이면 안 된다.
    title = resolve_title(cf, d, autonumber=not (args.no_autonumber or args.update))
    labels = [KIND_KO, f'{d["round"]}차']
    folder_id, folder_path, make_folder = resolve_folder(cf, d)

    template = cf.get_storage(CONFIG["guide"])
    fields = d.get("fields", {})
    body, missing = render(template, fields)
    check_links()

    unused = [k for k in fields if k not in field_labels(parse_storage(template))]
    if unused:
        print(f"경고: 가이드에 없는 항목이라 무시됨 — {', '.join(unused)}", file=sys.stderr)
    if missing:
        print(f"경고: 값이 없어 비워둔 항목 — {', '.join(missing)}", file=sys.stderr)

    if args.dry_run:
        base = build_title(d, None)
        if title != base:
            print(f"참고  : '{base}' 가 이미 있어 번호를 붙였다")
        if args.update:
            existing = cf.find_page(title)
            print(f"동작  : {'갱신 ' + existing if existing else '없어서 새로 생성'}")
        if make_folder:
            print("참고  : 폴더를 새로 만든다")
        print(preview(title, folder_path, labels, fields))
        print(f"본문  : {len(body)}자, 문서 안 링크 {len(_LINKS)}개 모두 짝이 맞음")
        if args.show_body:
            print(body)
        return

    existing = cf.find_page(title)
    if existing and not args.update:
        sys.exit(f"같은 제목의 페이지가 이미 있다: {title}\n"
                 "덮어쓰지 않는다. --update 로 갱신하거나 제목을 바꿀 것.")

    if existing:
        page = cf.update_page(existing, title, body)
        cf.add_labels(existing, labels)       # 라벨은 중복 추가해도 그대로다
        assign_owner(cf, existing, fields)
        print(f"갱신됨: {title}")
        print(f"버전  : {page['version']['number']}")
        print(f"주소  : {CONFIG['base_url']}/spaces/{CONFIG['space_key']}"
              f"/pages/{existing}")
        return

    if make_folder:
        folder_id = cf.create_folder(folder_id, d["new_folder"].strip())
    page = cf.create_page(folder_id, title, body)
    cf.add_labels(page["id"], labels)
    assign_owner(cf, page["id"], fields)

    print(f"만들어짐: {title}")
    print(f"폴더    : {folder_path} ({folder_id})")
    print(f"라벨    : {', '.join(labels)}")
    print(f"주소    : {CONFIG['base_url'] + page['_links']['webui']}")


if __name__ == "__main__":
    main()
