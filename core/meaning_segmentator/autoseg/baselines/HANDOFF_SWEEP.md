# 비교군 네이티브 노브 스윕을 다른 기계에서 돌리기

en→X 표의 비교군 곡선을 **각 정책 자기 노브로** 다시 만드는 작업이다. GPU 한 장과
하루 이틀이 필요하고 **API 비용은 0** 이다 (전부 로컬 모델).

## 왜

지금 곡선은 정책마다 라벨 한 벌을 만들어 두고 우리 노브 `T` 로 경계를 솎아 지연을
옮긴다. 축이 우리 것이라, 문헌이 하는 방식(자기 노브를 스윕해 곡선을 그린다)과 다르다.
노브가 있는 정책은 그 노브로 재야 비교가 대칭이 된다.

내부 NMT 도 평가 번역기와 같은 madlad 로 통일했다. AlignAtt 의 어텐션과 MU 의 접두사
일치는 둘 다 "이 모델이 지금 무엇을 아는가" 를 묻는데, 그 모델이 실제로 번역을 내놓는
모델과 다르면 엉뚱한 모델에 대해 판정하게 된다. 그래서 **기존 NLLB 산출(`alignatt_*`,
`mu_prefix_*`)은 새 점과 한 곡선에 못 섞는다** — f=2 와 n=10 도 다시 만드는 목록에 있다.

## 대상은 둘뿐이다

| 정책 | 노브 | 내부 모델 | 다시 만드나 |
|---|---|---|---|
| AlignAtt | `f` = 2, 4, 6, 8 | madlad | **예** |
| Prefix-match MU | `n_cands` = 2, 10, 50 | madlad | **예** |
| SASST | `max_chunk` | spaCy 파서 | 아니오 — 이미 있다 |
| Causal Align | 없음 | SimAlign + 참조 번역 | 아니오 — NMT 를 안 쓴다 |
| Punctuation | 없음 | 정규식 | 아니오 |

## 준비

```bash
git clone <저장소> && cd STiTy && git checkout autoseg-judge && git pull   # 5be29c4 이후
```

- 파이썬 환경에 `torch`, `transformers`, `sentencepiece`. 이 기계에서는 `.venv-autoseg`.
- `google/madlad400-3b-mt` 를 HuggingFace 에서 자동으로 받는다 (수 GB).
- **오디오는 필요 없다.** 소스 문장은 저장소에 든 매니페스트에서 읽는다
  (`evaluation/ast/manifests/covost2_en-{de,ja,zh}_full.jsonl`, 각 15,530줄).
- **VRAM 은 프로세스당 6.5GB.** 타깃별로 셋을 나란히 띄우려면 20GB 이상 비어 있어야 한다.
  `nvidia-smi` 로 먼저 확인할 것.

## 실행

반드시 tmux 로 띄운다. 타깃별로 나눠야 GPU 노는 틈이 메워진다 (배치가 1 이라 한 프로세스만
돌리면 사용률이 30% 에 그친다).

```bash
for t in de ja zh; do
  tmux new-session -d -s madlad-$t -c $(pwd) \
    "TGT=$t bash core/meaning_segmentator/tools/covost2_chain/24_native_madlad.sh"
done
```

`STAGE=alignatt` 또는 `STAGE=mu` 로 단계를 나눌 수 있다 (기본은 둘 다, AlignAtt 먼저).

진행은 로그로 본다.

```
core/meaning_segmentator/experiment/artifacts/en2x/covost2/full/logs/baselines/
  madlad_<타깃>.log                     단계별 시작·종료
  alignatt_mad_f<f>_<타깃>.log          ETA 줄
  mu_prefix_mad_n<n>_<타깃>.log
```

실측: AlignAtt 은 (f, 타깃) 하나에 단독 3.7시간. 12개를 셋씩 병렬로 돌리면 15~20시간이다.
MU 는 beam 이 붙어 더 걸린다.

## 주의

- **`--label` 을 주지 마라.** 스크립트가 안 준다. 기본값 `auto_best` 는 이 디렉토리에 없어서
  매니페스트 순서 15,530문장을 그대로 읽는데 그게 맞는 동작이다. `auto_run13_mg1` 을 주면
  옛 15,430 으로 되돌아간다.
- 중간에 죽어도 100건마다 `<라벨>.partial.jsonl` 에 흘려 쓰므로 같은 명령을 다시 돌리면
  이어진다. `--resume` 없이 돌리면 그 진행분을 **지운다** (스크립트는 항상 붙인다).
- 어텐션 층은 `Nmt` 가 모델을 보고 고른다 (madlad 26 / nllb 5). 바꿀 이유가 없으면 두어라.
  다시 재려면 `python -m core.meaning_segmentator.autoseg.baselines.probe_attn_layer`.

## 끝나면

산출물은 파일 21개, 각 약 4MB.

```
core/meaning_segmentator/experiment/artifacts/en2x/covost2/full/baselines/
  alignatt_mad_f{2,4,6,8}_{de,ja,zh}_test.json
  mu_prefix_mad_n{2,10,50}_{de,ja,zh}_test.json
```

확인 두 가지.

1. **행 수가 정확히 15,530** 이어야 한다.

   ```bash
   python - <<'EOF'
   import json, glob
   for p in sorted(glob.glob("core/meaning_segmentator/experiment/artifacts/en2x/covost2/full/baselines/*_mad_*_test.json")):
       d = json.load(open(p))
       print(len(d["rows"]), sum(len(r["pieces"]) for r in d["rows"]) / len(d["rows"]), p.split("/")[-1])
   EOF
   ```

2. **노브가 지연축을 만드는지**. 평균 조각 수가 AlignAtt 은 f 가 커질수록 **줄어야** 하고,
   MU 는 n 이 커질수록 **늘어야** 한다. 방향이 반대거나 값이 안 움직이면 포팅이 망가진
   것이므로 평가로 넘어가지 말고 알려 달라. 로그에도 "평균 조각수" 로 찍힌다.

그 21개 파일만 커밋해서 `autoseg-judge` 에 푸시하면 된다. **평가(`bleu_eval`)와 그림은
원래 기계에서 돌린다** — 번역 캐시가 거기 있어서 여기서 돌리면 15,530문장을 처음부터 다시
번역하게 된다.
