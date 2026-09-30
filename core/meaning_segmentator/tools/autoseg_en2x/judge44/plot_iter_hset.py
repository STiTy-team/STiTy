"""judge44 — 이터레이션이 쌓이면서 STAMP 이 얼마나 오르는가.

x축은 채택된 이터 수, y축은 v0 대비 STAMP 이득이다. 절대값이 아니라 이득으로 그리는
이유는 dev(500문장)와 test(560문장)의 v0 절대값이 다르기 때문이다 — 같은 축에 놓으려면
각자의 v0 를 0 으로 맞춰야 한다.

두 계열이 답하는 질문이 다르다.

  dev  이터마다 채택 판정에 쓴 쌍체 Δ 의 **누적합**. 후보 여섯 중 최선을 고르는 구조라
       위로 부풀어 있다 (run.sh 참조). 구간은 안 그린다 — 이터별 CI 를 더하려면 다섯 Δ 가
       독립이어야 하는데 같은 dev 500문장과 같은 기준선 벌 위에서 나온 값들이다.
  test 채택본을 홀드아웃 560문장 3벌로 다시 잰 값. 이터 3·4·5 에서만 쟀다.
       dev 에서 쌓인 이득이 잡음이었다면 여기서 사라진다.

실행:
    .venv-autoseg/bin/python core/meaning_segmentator/tools/autoseg_en2x/judge44/plot_iter_hset.py
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve()
while not (ROOT / ".git").exists():
    ROOT = ROOT.parent

INK, GRID = "#0b0b0b", "#d9d8d4"
GREEN, ORANGE = "#1a7a1a", "#e8712f"

# 로그와 산출물에 남은 이름은 `H_set` 이다 — 그림에만 STAMP 로 적는다
RE_ACCEPT = re.compile(r"\[iter (\d+)\] Δ H_set ([+\-][\d.]+) \[([+\-][\d.]+), ([+\-][\d.]+)\].*→ accept")
RE_FINAL = re.compile(
    r"\[최종\] test H_set ([\d.]+) / 오라클 ([\d.]+) / v0 ([\d.]+) Δ ([+\-][\d.]+) "
    r"\[([+\-][\d.]+), ([+\-][\d.]+)\]")


def parse_log(path):
    """로그를 순서대로 읽어 (채택 Δ, 홀드아웃 측정) 둘을 뽑는다.

    `--resume` 로 여러 번 이어 돌린 로그라 같은 이터가 여러 번 나온다. 채택은 이터 번호로
    덮어쓰고(마지막 판정이 유효), 홀드아웃은 **그 시점까지 채택된 이터 수**를 x 로 붙인다.
    """
    accepted, finals, last_iter = {}, [], 0
    for line in path.read_text(errors="replace").splitlines():
        m = RE_ACCEPT.search(line)
        if m:
            it = int(m.group(1))
            accepted[it] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
            last_iter = it
            continue
        m = RE_FINAL.search(line)
        if m:
            finals.append({
                "iter": last_iter,
                "h": float(m.group(1)), "oracle": float(m.group(2)), "v0": float(m.group(3)),
                "delta": float(m.group(4)), "lo": float(m.group(5)), "hi": float(m.group(6)),
            })
    # 같은 이터에서 두 번 잰 홀드아웃이 있으면 마지막 것만 남긴다
    return accepted, {f["iter"]: f for f in finals}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="en2x/en-multi/judge44")
    ap.add_argument("--log", default=None, help="기본값은 run 이름에서 유도")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    art = ROOT / "core/meaning_segmentator/experiment/artifacts"
    run_dir = art / a.run
    tag = a.run.rstrip("/").split("/")[-1]
    log = Path(a.log) if a.log else art / a.run.split("/")[0] / "logs" / f"{tag}.log"
    accepted, finals = parse_log(log)

    # 채택 Δ 는 history.json 이 정본이다. 로그는 홀드아웃 측정을 가져오는 데만 쓴다.
    hist = json.loads((run_dir / "history.json").read_text())
    adopted = {e["iter"]: e["delta"] for e in hist if e.get("adopted")}
    for it, d in adopted.items():
        accepted[it] = (d["mean"], d["lo"], d["hi"])

    iters = sorted(accepted)
    xs = [0] + iters
    cum, m = [0.0], 0.0
    for it in iters:
        m += accepted[it][0]
        cum.append(m)
    cum = np.array(cum)

    fx = sorted(finals)
    fy = np.array([finals[i]["delta"] for i in fx])
    flo = np.array([finals[i]["delta"] - finals[i]["lo"] for i in fx])
    fhi = np.array([finals[i]["hi"] - finals[i]["delta"] for i in fx])
    last = finals[fx[-1]]
    head = last["oracle"] - last["v0"]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "mathtext.fontset": "stix",
    })
    fig, ax = plt.subplots(figsize=(8.0, 3.6), dpi=200)
    ax.set_facecolor("#ffffff")
    ax.grid(True, color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK)
        ax.spines[s].set_linewidth(1.2)

    ax.plot(xs, cum, color=GREEN, linewidth=2.4, marker="o", markersize=8,
            markerfacecolor="white", markeredgewidth=2.0, markeredgecolor=GREEN,
            label="dev 500 (cumulative adopted $\Delta$)", zorder=3)
    ax.errorbar(fx, fy, yerr=[flo, fhi], color=ORANGE, linewidth=2.4,
                marker="^", markersize=9, markerfacecolor="white",
                markeredgewidth=2.0, markeredgecolor=ORANGE,
                capsize=5, capthick=1.8, elinewidth=1.6,
                label="test 560 holdout (3 draws, 95% CI)", zorder=4)

    for x, y in zip(fx, fy):
        # dev 곡선과 겹치지 않는 쪽에 적는다
        dev_here = cum[xs.index(x)] if x in xs else y
        dy = 9 if y > dev_here else -18
        ax.annotate(f"+{y:.4f}", xy=(x, y), xytext=(4, dy), textcoords="offset points",
                    fontsize=13, fontweight="bold", color=ORANGE)
    ax.annotate(f"+{cum[-1]:.4f}", xy=(xs[-1], cum[-1]), xytext=(4, 8),
                textcoords="offset points", fontsize=13, fontweight="bold", color=GREEN)

    leg = ax.legend(loc="upper left", frameon=False, fontsize=15,
                    handlelength=2.2, borderaxespad=0.2, labelspacing=0.3)
    for t in leg.get_texts():
        t.set_fontweight("bold")

    ax.set_xlabel("adopted iterations", fontsize=17)
    ax.set_ylabel("$\\Delta$ STAMP  vs v0", fontsize=17)
    ax.set_xticks(xs)
    ax.tick_params(labelsize=14)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.set_xlim(-0.25, xs[-1] + 0.45)

    ax.annotate(f"v0 baseline (test STAMP {last['v0']:.4f})",
                xy=(0.985, 0.04), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=16, fontweight="bold", color=INK)

    fig.tight_layout()
    out = Path(a.out) if a.out else run_dir / f"iter_hset_{tag}"
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(f"dev 누적 {[round(c, 4) for c in cum]}")
    print(f"test {[(i, finals[i]['delta']) for i in fx]}")
    print(f"→ {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
