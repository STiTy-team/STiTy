#!/bin/bash
# judge02 의 최종 프롬프트로 CoVoST2 X→en 을 라벨링하고 점수 임계값 스윕으로 평가한다.
#
# 무엇이 en2x 판(13_judge44_label.sh + 14b_judge44_eval.sh)과 다른가
#   프롬프트 하나   채택본이 있으면 best_prompt.txt, 없으면 prompt_v0.txt 를 쓴다. v0 대 채택본
#                  비교를 하지 않으므로 라벨링이 언어당 한 벌이다.
#   점수 격자만     지연 노브를 --score-grid 로만 낸다(--no-auto-t). 비교군을 안 돌리므로
#                  --t-grid 가 쓰이는 데가 없어 기본값 그대로 둔다. 비교군을 다시 넣을 때는
#                  T 격자를 같이 줘야 한다 — 안 주면 비교군이 한 점으로 줄어 곡선이 사라진다.
#   소스가 영어가 아니다  --spaced / --src-spaced 가 언어마다 다르다. de 는 어절, zh·ja 는 글자다.
#
# --punct-attach 로 구두점만 있는 조각을 앞 조각에 붙인 조건(auto_S<th>_p)을 나란히 낸다. 띄어쓰기 없는
# 언어에서 라벨이 문말 부호 앞에 거의 항상 높은 점수를 줘, ja 는 임계값 95 에서 문장의 92% 가 "。"
# 한 글자 조각으로 끝났다. 붙이면 같은 지연에서 COMET 이 약 0.03 오른다. de 는 3,000문장 중 2문장만 바뀐다.
# 격자의 95~98 은 그 조건에서 곡선이 실제로 움직이는 구간이라 촘촘히 둔다.
#
# --tag-order score 를 반드시 준다. 기본값 rank 로 돌리면 <SEG:1> 을 최고 확신으로 읽어
# 절단이 거꾸로 돈다.
#
# 스모크 게이트를 먼저 통과해야 본런이 돈다. 30문장으로 format_pass 와 text_preserved 를 보고,
# 형식이 깨진 프롬프트로 3,000문장을 태우는 것을 막는다.
#
# 비용은 라벨링에만 든다 — 번역은 로컬 madlad, COMET 도 로컬 GPU 라 API 비용이 0 이다.
#
#   SRC=de bash core/meaning_segmentator/tools/covost2_chain/34_x2en_judge02.sh
#   SMOKE_ONLY=1 SRC=de bash ...      # 스모크만
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LANGSMITH_TRACING=0
PY=${PY:-.venv-autoseg/bin/python}
L=${SRC:?SRC=de|zh|ja}
RUN=${RUN:-judge02}
A=core/meaning_segmentator/experiment/artifacts
# 본런 예산은 **이번 스모크의 실측 비용**으로 잡는다 — 문장당 비용 × N × 2 와 아래 BUD 중 큰 쪽.
# 고정값만 두면 프롬프트가 바뀔 때 어긋난다. 옛 v0 프롬프트의 de 스모크는 30문장 $0.0436 였는데
# judge02 best_prompt 는 $0.1136 으로 2.6배였고, 그래서 BUD=10 이 3,000문장 본런($11.3)을 죽였다.
# BUD 는 스모크가 캐시로 공짜로 지나가 비용을 못 잴 때(재실행)의 하한이다. BUDGET_MAIN 을 주면
# 그 값을 그대로 쓴다.
case $L in
  de) N=3000; SPACED=yes; SP=1; BUD=18 ;;
  zh) N=3000; SPACED=no;  SP=0; BUD=18 ;;
  ja) N=678;  SPACED=no;  SP=0; BUD=6  ;;
  *)  echo "!! 모르는 언어: $L"; exit 1 ;;
esac
J=$A/x2en/$L-multi/$RUN
# 런 디렉토리를 run02 과 **분리한다.** bleu_eval 과 to_prompt_eval 의 산출 이름은 타깃·라벨로만
# 정해져서(bleu/en.json, prompt_eval/<label>_test.json) 같은 run-id 로 돌리면 run02 의 결과를
# 덮어쓴다. 번역 캐시만 심볼릭 링크로 공유한다 — 조각이 겹치는 만큼 madlad 를 다시 안 돌린다.
SRCD=$A/x2en/covost2/$L-en_n$N
D=$A/x2en/covost2/${L}-en_n${N}_$RUN
MAN=evaluation/ast/manifests/covost2_$L-en_n$N.jsonl
RID=x2en/covost2/${L}-en_n${N}_$RUN
LABEL=auto_$RUN
SGRID="${SGRID:-5 10 20 40 60 70 80 90 95 96 97 98 99}"
W=${W:-240}
BUDGET_MAIN_ENV=${BUDGET_MAIN:-}
LOG=$D/logs/${RUN}_chain.log
mkdir -p $D/labels $D/logs $D/cache
[ -e $D/mt_cache ] || ln -s ../$L-en_n$N/cache $D/mt_cache
ts () { date '+%F %T'; }

# **이미 끝난 언어는 다시 돌리지 않는다.** 같은 언어를 두 곳에서 큐에 걸었을 때 평가가 두 번
# 돌아 GPU 시간을 버리고, 라벨링이 캐시를 빗나가면 돈까지 다시 나간다.
if [ -s $D/$RUN.done ]; then
  echo "== $(ts) $L 이미 완료 ($(cat $D/$RUN.done)) — 건너뛴다" >> $LOG
  exit 0
fi

# 채택본이 있으면 그것, 없으면 v0. 두 이터가 다 기각이면 best_prompt.txt 가 안 생긴다.
if [ -s $J/best_prompt.txt ]; then PROMPT=$J/best_prompt.txt; WHICH=best
else                               PROMPT=$J/prompt_v0.txt;   WHICH=v0
fi
[ -s "$PROMPT" ] || { echo "!! 프롬프트 없음: $PROMPT" | tee -a $LOG; exit 1; }
echo "== $(ts) $L 시작 — 프롬프트 $WHICH ($PROMPT), 문장 $N, 띄어쓰기 $SPACED" >> $LOG

# 로그는 **이어 쓴다.** 죽은 실행의 비용 기록(`진행 … 누적 비용`)이 재실행에 덮이면 그 지출을
# 복원할 길이 없다. 요약 JSON 은 그래서 마지막 것을 읽는다.
label () {   # <이름> <limit인자> <예산>
  $PY -u -m core.meaning_segmentator.tools.covost2_label.label_covost2 \
    --provider openai --model gpt-5-mini \
    --prompt "$PROMPT" --manifest $MAN --spaced $SPACED \
    --out $D/labels/$1.jsonl --cache $D/cache/$1.cache.json \
    --min-gap 1 --t-floor 2 --batch-size 6 --workers $W $2 \
    --max-tokens 24000 --timeout 420 --budget $3 --cache-every 1 \
    >> $D/logs/$1.log 2>&1
}

gate () {    # <이름> — 형식이 깨지면 본런을 시작하지 않는다
  $PY - "$D/logs/$1.log" <<'PYGATE'
import json, re, sys
t = open(sys.argv[1]).read()
ms = re.findall(r"\{[^{}]*\"format_pass\".*?\}", t, re.S)
if not ms:
    print("  요약 JSON 없음"); sys.exit(1)
d = json.loads(ms[-1])
ok = d["format_pass"] >= 0.90 and d["text_preserved"] >= 0.95
print(f"  format_pass {d['format_pass']} / text_preserved {d['text_preserved']} / "
      f"first_pass {d['first_pass']} / {d['wall_sec']}초 → {'통과' if ok else '불통과'}")
sys.exit(0 if ok else 1)
PYGATE
}

NAME=${RUN}_${WHICH}
# 본런 라벨이 이미 다 차 있으면 스모크도 본런도 건너뛴다.
if [ -s $D/labels/$NAME.jsonl ] && [ "$(wc -l < $D/labels/$NAME.jsonl)" -eq "$N" ]; then
  echo "== $(ts) $L 라벨 $N줄 이미 있음 — 라벨링 건너뛴다" >> $LOG
  SKIP_LABEL=1
else
  SKIP_LABEL=
fi

if [ -z "$SKIP_LABEL" ]; then
echo "===== $(ts) $L 스모크 30문장 =====" | tee -a $LOG
label "${NAME}_smoke" "--limit 30" 2.0
if ! gate "${NAME}_smoke" | tee -a $LOG; then
  echo "!! $L 스모크 불통과 — 중단" | tee -a $LOG; exit 1
fi
[ -n "${SMOKE_ONLY:-}" ] && { echo "  SMOKE_ONLY — 여기서 멈춘다" | tee -a $LOG; exit 0; }

if [ -n "$BUDGET_MAIN_ENV" ]; then
  BUDGET_MAIN=$BUDGET_MAIN_ENV
else
  BUDGET_MAIN=$($PY - "$D/logs/${NAME}_smoke.log" $N $BUD <<'PYBUD'
import json, re, sys
t = open(sys.argv[1]).read(); n_all = int(sys.argv[2]); floor = float(sys.argv[3])
ms = re.findall(r"\{[^{}]*\"format_pass\".*?\}", t, re.S)
d = json.loads(ms[-1]) if ms else {}
per = d.get("cost", 0) / max(1, d.get("n", 1))
print(f"{max(floor, per * n_all * 2):.2f}")
PYBUD
)
fi
echo "===== $(ts) $L 본런 ${N}문장 (예산 \$$BUDGET_MAIN) =====" >> $LOG
label "$NAME" "" $BUDGET_MAIN
rc=$?
echo "== $(ts) $L 라벨링 exit=$rc" >> $LOG
grep -E "^\{" $D/logs/$NAME.log | tail -1 >> $LOG
[ $rc -eq 0 ] || exit 1
fi

if [ -s $D/prompt_eval/${LABEL}_test.json ]; then
  echo "== $(ts) $L 변환 결과 있음 — 건너뛴다" >> $LOG
else
echo "== $(ts) $L 변환 (tag-order score, min-gap 1)" >> $LOG
$PY -u core/meaning_segmentator/tools/covost2_label/to_prompt_eval.py \
  --labels $D/labels/$NAME.jsonl --run-id $RID --label $LABEL --split test \
  --min-gap 1 --tag-order score --spaced $SPACED \
  >> $D/logs/to_prompt_eval.log 2>&1 || { echo "!! 변환 실패" >> $LOG; exit 1; }
fi

if [ -s $D/bleu/en.json ]; then
  echo "== $(ts) $L bleu 결과 있음 — 건너뛴다" >> $LOG
else
echo "== $(ts) $L bleu_eval (점수 격자 $SGRID)" >> $LOG
$PY -u -m core.meaning_segmentator.autoseg.scoring.bleu_eval \
  --run-id $RID --label $LABEL --split test \
  --dataset covost2 --src $L --manifest-tag n$N --targets en \
  --score-grid $SGRID --punct-attach --no-auto-t --src-spaced $SP \
  --translate-engine local --local-mt-model google/madlad400-3b-mt --mt-batch 48 \
  --workers 24 --bootstrap 0 --no-sentence-bleu --no-auto-greedy \
  > $D/logs/${RUN}_bleu_eval.log 2>&1
rc=$?
echo "== $(ts) $L bleu_eval exit=$rc" >> $LOG
# 여기서 끊는다. 실패한 채 COMET 으로 넘어가면 빈 조건 위에서 점수가 나와, 로그만 보면
# 성공처럼 보이는 산출물이 남는다.
[ $rc -eq 0 ] || exit 1
fi

echo "== $(ts) $L comet" >> $LOG
$PY -u -m core.meaning_segmentator.autoseg.baselines.comet_score \
  --run-id $RID --dataset covost2 --manifest-tag n$N --src $L \
  --label $LABEL --split test --targets en --only-missing \
  --model Unbabel/wmt22-comet-da --batch-size 64 > $D/logs/${RUN}_comet.log 2>&1
rc=$?
echo "== $(ts) $L comet exit=$rc" >> $LOG
[ $rc -eq 0 ] || exit 1
date '+%F %T' > $D/${RUN}.done
echo "== $(ts) $L 전부 완료" >> $LOG
