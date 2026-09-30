"""DialogueContext 번역 실행기. 규칙은 DESIGN.md (정준 요청, 프롬프트, 지연 측정, 결과 행).

    PYTHONPATH=<repo> <venv-tf5>/bin/python -u evaluation/DialogueContext/scripts/run_translations.py \
        --config evaluation/DialogueContext/configs/experiment.yml --phase main

--phase main   주 실험 12 조건 + NONE 기준선. 전 모델·전 조건을 시드로 섞어 한 줄로 차례로 돈다.
--phase s1     보조 실험. 모델마다 대화를 처음부터 번역하며 자기 출력을 앞 번역으로 쓴다.
--phase all    main 다음 s1 을 같은 프로세스에서.

결과는 results/<run_id>/translations.jsonl 에 행마다 바로 붙는다. 다시 띄우면 끝난 job_id 는 건너뛴다.
실패한 job 은 다시 띄울 때 다시 돈다. 같은 job_id 행이 여러 개면 마지막 행이 유효하다.
"""
import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import yaml  # noqa: E402

from core.utils import env  # noqa: E402
from dctx import context as C  # noqa: E402
from dctx.ledger import CostLedger, append, now, read_rows  # noqa: E402

DESIGN = HERE.parent / "DESIGN.md"


CUDA_RETRIES = 3
CUDA_WAIT_SEC = 30
MAX_FAIL_STREAK = 5


def gpu_used_mib() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return "?"


class BudgetStop(RuntimeError):
    pass


# ── 데이터 ────────────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load_data(data_dir: Path, limit: int | None) -> tuple[list[dict], dict]:
    instances = read_jsonl(data_dir / "instances.jsonl")
    if limit:
        instances = instances[:limit]
    dialogues = {}
    path = data_dir / "dialogues.jsonl"
    if path.exists():
        for turn in read_jsonl(path):
            dialogues.setdefault(turn["dialogue_id"], []).append(turn)
        for turns in dialogues.values():
            turns.sort(key=lambda t: t["turn_id"])
    return instances, dialogues


def check_instance(inst: dict, dialogues: dict) -> None:
    turns = inst["previous_turns"]
    ids = [t["turn_id"] for t in turns]
    if ids != sorted(ids) or (ids and ids[-1] >= inst["target_turn_id"]):
        raise ValueError(f"{inst['instance_id']}: previous_turns not oldest-first before target")
    if len(turns) < 5:
        print(f"[warn] {inst['instance_id']}: only {len(turns)} previous turns", flush=True)
    dialogue = dialogues.get(inst["dialogue_id"])
    if dialogue:
        by_id = {t["turn_id"]: t for t in dialogue}
        target = by_id.get(inst["target_turn_id"])
        if target is None or target["ko"] != inst["current_source"]:
            raise ValueError(f"{inst['instance_id']}: current_source differs from dialogues.jsonl")


def labels_for(inst: dict, dialogues: dict) -> dict:
    turns = dialogues.get(inst["dialogue_id"])
    if not turns:
        turns = inst["previous_turns"] + [
            {"turn_id": inst["target_turn_id"], "speaker": inst["current_speaker"]}]
    return C.speaker_labels(turns)


# ── 작업 목록 ─────────────────────────────────────────────────────────────────

def parse_conditions(text: str | None) -> set | None:
    if not text:
        return None
    out = set()
    for part in text.split(","):
        strategy, n = part.strip().split(":")
        out.add((strategy, int(n)))
    return out


def main_jobs(cfg: dict, instances: list[dict], models: list[str], only: set | None) -> list[dict]:
    conditions = [(s, n) for s in cfg["strategies"] for n in cfg["context_n"]]
    conditions.append((cfg["baseline"]["strategy"], cfg["baseline"]["n"]))
    jobs = []
    for model in models:
        for inst in instances:
            for strategy, n in conditions:
                if only and (strategy, n) not in only:
                    continue
                jobs.append({
                    "job_id": f"{model}|{inst['instance_id']}|{strategy}|{n}",
                    "phase": "baseline" if strategy == C.BASELINE else "main",
                    "model": model, "inst": inst, "strategy": strategy, "n": n,
                })
    jobs.sort(key=lambda j: j["job_id"])
    random.Random(cfg["seed"]).shuffle(jobs)
    for i, job in enumerate(jobs):
        job["order_index"] = i
    return jobs


def s1_chains(cfg: dict, instances: list[dict], dialogues: dict, models: list[str]) -> list[dict]:
    targets = {}
    for inst in instances:
        targets.setdefault(inst["dialogue_id"], {})[inst["target_turn_id"]] = inst
    missing = [d for d in targets if d not in dialogues]
    if missing:
        raise SystemExit(f"S1 needs dialogues.jsonl; missing dialogues {missing}")
    chains = []
    for model in models:
        for strategy in cfg["s1"]["strategies"]:
            for dialogue_id in sorted(targets):
                last = max(targets[dialogue_id])
                chains.append({"model": model, "strategy": strategy, "n": cfg["s1"]["n"],
                               "dialogue_id": dialogue_id,
                               "turns": [t for t in dialogues[dialogue_id] if t["turn_id"] <= last],
                               "targets": targets[dialogue_id]})
    chains.sort(key=lambda c: (c["model"], c["strategy"], c["dialogue_id"]))
    random.Random(cfg["seed"]).shuffle(chains)
    return chains


# ── 요청 → 행 ────────────────────────────────────────────────────────────────

def build(cfg: dict, strategy: str, n: int, utterance: str, previous: list[dict], labels: dict,
          speaker: str) -> tuple[dict, str, str, str]:
    request = C.canonical_request(
        source_language=cfg["source_language"], target_language=cfg["target_language"],
        utterance=utterance, strategy=strategy, n=n, previous=previous, labels=labels,
        current_speaker=speaker)
    return request, C.SYSTEM, C.user_prompt(request), C.context_block(request)


class Runner:
    def __init__(self, cfg: dict, out: Path, backends: dict, ledger: CostLedger,
                 latency_valid: bool, session: str):
        self.cfg, self.out, self.backends, self.ledger = cfg, out, backends, ledger
        self.latency_valid = latency_valid
        self.session = session
        self.rows_path = out / "translations.jsonl"
        self.budget_hit = set()
        self.streak = {}

    def call(self, model: str, request: dict, system: str, user: str, ctx: str, tag: dict) -> dict:
        backend = self.backends[model]
        if backend.kind == "local":
            return self.local(backend, system, user, request)
        if self.ledger.over_budget():
            raise BudgetStop(f"budget_usd {self.ledger.budget} reached "
                             f"(estimated ${self.ledger.spent():.4f})")
        tag = {"api_model": model, **tag}
        if backend.kind == "deepl":
            return backend.translate(system, user, request, tag=tag, context=ctx)
        return backend.translate(system, user, request, tag=tag)

    def local(self, backend, system: str, user: str, request: dict) -> dict:
        """GPU 를 다른 세션과 나눠 쓰므로 CUDA 메모리 오류는 잠시 기다렸다 다시 한다."""
        import torch

        for attempt in range(1, CUDA_RETRIES + 2):
            try:
                res = backend.translate(system, user, request)
                res["attempts"] = attempt
                return res
            except (torch.OutOfMemoryError, RuntimeError) as e:
                if "CUDA" not in str(e) and not isinstance(e, torch.OutOfMemoryError):
                    raise
                if attempt > CUDA_RETRIES:
                    raise
                print(f"[cuda] {backend.name}: {str(e)[:120]}; retry {attempt} in "
                      f"{CUDA_WAIT_SEC}s (gpu used {gpu_used_mib()} MiB)", flush=True)
                torch.cuda.empty_cache()
                time.sleep(CUDA_WAIT_SEC)

    def run(self, *, job_id: str, phase: str, order_index: int, model: str, request: dict,
            system: str, user: str, ctx: str, reference: str | None, meta: dict) -> dict | None:
        is_deepl = self.backends[model].kind == "deepl"
        started_at = now()
        error, res = None, {}
        try:
            res = self.call(model, request, system, user, ctx,
                            {"job_id": job_id, "phase": phase, "session": self.session})
        except BudgetStop as e:
            if model not in self.budget_hit:
                self.budget_hit.add(model)
                print(f"[budget] {e}; skipping further {model} calls", flush=True)
            return None
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        self.streak[model] = self.streak.get(model, 0) + 1 if error else 0
        raw = res.get("raw_output")
        hypothesis, violation = C.clean_output(raw) if raw is not None else ("", None)
        if res.get("truncated"):
            violation = "; ".join(filter(None, [violation, "truncated_max_tokens"]))
        row = {
            "job_id": job_id,
            "phase": phase,
            "order_index": order_index,
            "model": model,
            **meta,
            "context_strategy": request["context_strategy"],
            "context_n": request["context_n"],
            "request": request,
            "prompt_system": None if is_deepl else system,
            "prompt_user": ctx if is_deepl else user,
            "current_source": request["current_utterance"],
            "reference": reference,
            "raw_output": raw,
            "hypothesis": hypothesis if raw is not None else None,
            "format_violation": violation if raw is not None else None,
            "latency_ms": res.get("latency_ms"),
            "ttft_ms": res.get("ttft_ms"),
            "generation_ms": res.get("generation_ms"),
            "input_tokens": res.get("input_tokens"),
            "output_tokens": res.get("output_tokens"),
            "input_chars": (len(request["current_utterance"]) + len(ctx)) if is_deepl
            else len(system) + len(user),
            "context_chars": len(ctx),
            "output_chars": len(hypothesis) if raw is not None else None,
            "error": error,
            "started_at": started_at,
            "latency_valid": self.latency_valid and phase != "warmup",
            "attempts": res.get("attempts"),
            "api_meta": res.get("api_meta"),
            "session": self.session,
        }
        append(self.rows_path, row)
        if self.streak[model] >= MAX_FAIL_STREAK:
            raise SystemExit(f"{model}: {MAX_FAIL_STREAK} failures in a row, last: {error}. "
                             f"Stopping; rerun resumes and retries failed jobs.")
        return row


def show(row: dict, i: int, total: int) -> None:
    hyp = (row["hypothesis"] or row["error"] or "")[:70].replace("\n", " ")
    ttft = f"{row['ttft_ms']:.0f}" if row["ttft_ms"] is not None else "-"
    lat = f"{row['latency_ms']:.0f}" if row["latency_ms"] is not None else "-"
    print(f"[{i}/{total}] {row['phase']:8s} {row['model']:13s} "
          f"{row['context_strategy']:11s} n={row['context_n']} "
          f"{row.get('instance_id') or row.get('dialogue_id')} lat={lat}ms ttft={ttft}ms "
          f"tok={row['input_tokens']}/{row['output_tokens']} "
          f"{'VIOL[' + row['format_violation'] + '] ' if row['format_violation'] else ''}| {hyp}",
          flush=True)


# ── 단계 ─────────────────────────────────────────────────────────────────────

def warmup(runner: Runner, cfg: dict, models: list[str]) -> None:
    sentences = cfg["warmup"]["sentences"]
    turns = [{"turn_id": i, "speaker": "AB"[i % 2], "ko": s["ko"], "en": s["en"]}
             for i, s in enumerate(sentences)]
    labels = C.speaker_labels(turns)
    kinds = [*cfg["strategies"], C.BASELINE]
    for model in models:
        count = cfg["warmup"]["local" if runner.backends[model].kind == "local" else "api"]
        for k in range(count):
            i = 1 + k % (len(turns) - 1)
            strategy = kinds[k % len(kinds)]
            n = 0 if strategy == C.BASELINE else min(i, 3)
            request, system, user, ctx = build(cfg, strategy, n, turns[i]["ko"], turns[:i],
                                               labels, turns[i]["speaker"])
            row = runner.run(job_id=f"warmup|{model}|{k}|{runner.session}", phase="warmup",
                             order_index=-1, model=model, request=request, system=system,
                             user=user, ctx=ctx, reference=turns[i]["en"],
                             meta={"instance_id": None, "dialogue_id": None,
                                   "target_turn_id": None})
            if row:
                show(row, k + 1, count)


def done_ids(path: Path) -> dict:
    """끝난 job_id. 마지막 행이 실패면 끝나지 않은 것으로 보고 다시 돈다."""
    done = {}
    for row in read_rows(path):
        if row.get("phase") == "warmup":
            continue
        if row.get("error"):
            done.pop(row["job_id"], None)
        else:
            done[row["job_id"]] = row
    return done


def run_main(runner: Runner, cfg: dict, instances: list[dict], dialogues: dict,
             models: list[str], args) -> None:
    jobs = main_jobs(cfg, instances, models, parse_conditions(args.conditions))
    done = done_ids(runner.rows_path)
    todo = [j for j in jobs if j["job_id"] not in done]
    print(f"[main] {len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run",
          flush=True)
    for i, job in enumerate(todo, 1):
        inst = job["inst"]
        request, system, user, ctx = build(
            cfg, job["strategy"], job["n"], inst["current_source"], inst["previous_turns"],
            labels_for(inst, dialogues), inst["current_speaker"])
        row = runner.run(job_id=job["job_id"], phase=job["phase"],
                         order_index=job["order_index"], model=job["model"], request=request,
                         system=system, user=user, ctx=ctx,
                         reference=inst["reference_translation"],
                         meta={"instance_id": inst["instance_id"],
                               "dialogue_id": inst["dialogue_id"],
                               "target_turn_id": inst["target_turn_id"]})
        if row:
            show(row, i, len(todo))


def run_s1(runner: Runner, cfg: dict, instances: list[dict], dialogues: dict,
           models: list[str], args) -> None:
    chains = s1_chains(cfg, instances, dialogues, models)
    done = done_ids(runner.rows_path)
    total = sum(len(c["turns"]) for c in chains)
    print(f"[s1] {len(chains)} chains, {total} turns", flush=True)
    order, step = 0, 0
    for chain in chains:
        model, strategy, n = chain["model"], chain["strategy"], chain["n"]
        labels = C.speaker_labels(chain["turns"])
        own = []
        for turn in chain["turns"]:
            order += 1
            job_id = f"s1|{model}|{strategy}|{chain['dialogue_id']}|t{turn['turn_id']:03d}"
            inst = chain["targets"].get(turn["turn_id"])
            if job_id in done:
                row = done[job_id]
            else:
                request, system, user, ctx = build(cfg, strategy, n, turn["ko"], own, labels,
                                                   turn["speaker"])
                row = runner.run(job_id=job_id, phase="s1", order_index=order, model=model,
                                 request=request, system=system, user=user, ctx=ctx,
                                 reference=turn["en"],
                                 meta={"instance_id": inst["instance_id"] if inst else None,
                                       "dialogue_id": chain["dialogue_id"],
                                       "target_turn_id": turn["turn_id"],
                                       "is_target": inst is not None})
                step += 1
                if row is None:
                    break  # 이 모델은 예산 초과. 다른 모델의 사슬은 계속 돈다
                show(row, step, total)
            if row.get("error"):
                print(f"[s1] chain {model}/{strategy}/{chain['dialogue_id']} stopped at turn "
                      f"{turn['turn_id']}: {row['error']}", flush=True)
                break
            own.append({"turn_id": turn["turn_id"], "speaker": turn["speaker"],
                        "src": turn["ko"], "tgt": row["hypothesis"]})


# ── 기록물 ───────────────────────────────────────────────────────────────────

def gpu_snapshot(out: Path, session: str, label: str) -> None:
    gpu = out / "gpu"
    gpu.mkdir(parents=True, exist_ok=True)
    try:
        text = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
    except Exception as e:  # noqa: BLE001
        text = f"nvidia-smi failed: {e!r}"
    (gpu / f"nvidia-smi_{session}_{label}.txt").write_text(text, encoding="utf-8")


def design_block(title: str) -> str:
    text = DESIGN.read_text(encoding="utf-8")
    m = re.search(rf"^{re.escape(title)}\n```\n(.*?)\n```", text, re.MULTILINE | re.DOTALL)
    if not m:
        raise SystemExit(f"DESIGN.md: block {title!r} not found")
    return m.group(1)


def check_design() -> None:
    """코드의 프롬프트가 DESIGN.md 의 문구와 글자 하나까지 같은지 본다."""
    if design_block("system:") != C.SYSTEM:
        raise SystemExit("SYSTEM prompt differs from DESIGN.md")
    turns = [
        {"turn_id": 0, "speaker": "A", "ko": "양말 말하는 거 맞지?", "en": "You mean the socks, right?"},
        {"turn_id": 1, "speaker": "B", "ko": "응. 그런데 집에 두고 왔어.",
         "en": "Yeah. But I left them at home."},
    ]
    request = C.canonical_request(
        source_language="ko", target_language="en", utterance="그래서 다시 가지러 갔지.",
        strategy="SPK_SRC_TGT", n=2, previous=turns, labels=C.speaker_labels(turns),
        current_speaker="A")
    if design_block("user (예: SPK_SRC_TGT, n=2):") != C.user_prompt(request):
        raise SystemExit("user prompt differs from DESIGN.md example")


def write_artifacts(out: Path, config_path: Path, session: str, data_dir: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    # 채점기는 run 디렉터리의 instances.jsonl 을 먼저 본다. 실행에 쓴 데이터를 그대로 남긴다.
    for name in ("instances.jsonl", "dialogues.jsonl"):
        saved, src = out / name, data_dir / name
        if not saved.exists():
            shutil.copy(src, saved)
        elif saved.read_bytes() != src.read_bytes():
            raise SystemExit(f"{src} differs from the copy in {out}; the data changed since this "
                             f"run started. Use a new --run-id.")
    saved = out / "config.yml"
    if not saved.exists():
        shutil.copy(config_path, saved)
    elif saved.read_bytes() != config_path.read_bytes():
        shutil.copy(config_path, out / f"config.{session}.yml")
        print(f"[warn] config changed since the run started; saved as config.{session}.yml",
              flush=True)
    (out / "prompt_template.txt").write_text(
        "# system (Qwen, Gemma, GPT)\n" + C.SYSTEM + "\n\n# user\n" + C.USER_TEMPLATE
        + "\n\n# notes\n"
        "{context}: one item per previous turn, oldest first. [Speaker k] line only for "
        "SPK_SRC_TGT, then 'SRC: ...' and/or 'TGT: ...'. '(none)' when there is no context.\n"
        "{speaker}: ' (Speaker k)' only for SPK_SRC_TGT, otherwise empty.\n"
        "DeepL: text = current utterance, context = the {context} string (omitted when empty).\n",
        encoding="utf-8")


# ── 백엔드 ───────────────────────────────────────────────────────────────────

LOAD_RETRIES = 60
LOAD_WAIT_SEC = 60


def load_with_retry(backend) -> float:
    """다른 세션이 GPU 를 잠깐 다 쓰고 있으면 로드가 CUDA OOM 으로 죽는다. 남의 프로세스는
    건드리지 않고, 비워질 때까지 기다렸다 다시 올린다. 기다린 시간은 로드 시간에 넣지 않는다."""
    import gc

    import torch

    for attempt in range(1, LOAD_RETRIES + 2):
        try:
            return backend.load()
        except torch.OutOfMemoryError as e:
            if attempt > LOAD_RETRIES:
                raise
            backend._model = None
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[load] {backend.name}: CUDA OOM while loading ({str(e)[:100]}); gpu used "
                  f"{gpu_used_mib()} MiB; retry {attempt}/{LOAD_RETRIES} in {LOAD_WAIT_SEC}s",
                  flush=True)
            time.sleep(LOAD_WAIT_SEC)


def make_backends(cfg: dict, models: list[str], ledger: CostLedger, out: Path,
                  session: str) -> dict:
    from dctx.backends import DeepL, LocalModel, OpenAIChat

    backends = {}
    for name in models:
        spec = cfg["models"][name]
        if spec["kind"] == "local":
            backend = LocalModel(name, spec["model"], spec.get("quant", "4bit"),
                                 cfg["generation"]["max_new_tokens"])
            seconds = load_with_retry(backend)
            import torch

            line = {"at": now(), "session": session, "model": name, "hf_id": spec["model"],
                    "quant": spec.get("quant"), "load_sec": round(seconds, 2),
                    "cuda_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
                    "transformers": __import__("transformers").__version__}
            append(out / "model_load.jsonl", line)
            print(f"[load] {name}: {seconds:.1f}s, allocated {line['cuda_allocated_gib']} GiB "
                  f"(total on this process)", flush=True)
        else:
            key = env.get(spec["key_env"])
            if not key:
                raise SystemExit(f"{name}: {spec['key_env']} missing in environment / .env")
            timeout = cfg.get("api_timeout_sec", 60)
            if spec["kind"] == "deepl":
                backend = DeepL(name, key, ledger, model_type=spec["model_type"],
                                timeout=timeout, max_retries=spec.get("max_retries", 5))
            elif spec["kind"] == "openai":
                backend = OpenAIChat(name, spec["model"], key, cfg["prices"][spec["model"]],
                                     ledger, reasoning_effort=spec.get("reasoning_effort", "none"),
                                     temperature=spec.get("temperature", 0),
                                     max_completion_tokens=spec.get("max_completion_tokens"),
                                     timeout=timeout, max_retries=spec.get("max_retries", 3))
            else:
                raise SystemExit(f"{name}: unknown kind {spec['kind']!r}")
        backends[name] = backend
    return backends


def show_prompts(cfg: dict, instances: list[dict], dialogues: dict) -> None:
    inst = instances[0]
    labels = labels_for(inst, dialogues)
    for strategy, n in [(s, 3) for s in cfg["strategies"]] + [(C.BASELINE, 0)]:
        request, system, user, ctx = build(cfg, strategy, n, inst["current_source"],
                                           inst["previous_turns"], labels, inst["current_speaker"])
        print(f"\n===== {inst['instance_id']} {strategy} n={n} — user prompt =====\n{user}")
        print(f"----- DeepL context -----\n{ctx or '(omitted)'}")
    print(f"\n===== system =====\n{C.SYSTEM}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default=str(HERE.parent / "configs" / "experiment.yml"))
    p.add_argument("--phase", choices=("main", "s1", "all"), default="main")
    p.add_argument("--limit-instances", type=int)
    p.add_argument("--models", help="comma list; default all in config")
    p.add_argument("--conditions", help="e.g. SRC:1,SPK_SRC_TGT:3,NONE:0 (main only; default all)")
    p.add_argument("--run-id")
    p.add_argument("--results-dir", help="base dir; results go to <dir>/<run_id>/")
    p.add_argument("--data-dir")
    p.add_argument("--parallel", action="store_true",
                   help="other translation processes share the GPU: mark rows latency_valid=false")
    p.add_argument("--show-prompts", action="store_true", help="print prompts for one instance")
    p.add_argument("--dry-run", action="store_true", help="build job list only; no model calls")
    args = p.parse_args()

    env.load(REPO / ".env")
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_id = args.run_id or cfg["run_id"]
    out = Path(args.results_dir or REPO / cfg["results_dir"]) / run_id
    data_dir = Path(args.data_dir or REPO / cfg["data_dir"])
    models = args.models.split(",") if args.models else list(cfg["models"])
    unknown = [m for m in models if m not in cfg["models"]]
    if unknown:
        raise SystemExit(f"unknown models {unknown}")
    if args.phase in ("main", "all") and args.parallel:
        raise SystemExit("--parallel is for s1 only; the main phase must run with concurrency 1")

    check_design()
    instances, dialogues = load_data(data_dir, args.limit_instances)
    for inst in instances:
        check_instance(inst, dialogues)
    print(f"[data] {len(instances)} instances, {len(dialogues)} dialogues from {data_dir}",
          flush=True)
    if args.show_prompts:
        show_prompts(cfg, instances, dialogues)
    if args.dry_run:
        jobs = main_jobs(cfg, instances, models, parse_conditions(args.conditions))
        print(f"[dry-run] main+baseline jobs: {len(jobs)}")
        if dialogues:
            chains = s1_chains(cfg, instances, dialogues, models)
            print(f"[dry-run] s1 chains: {len(chains)}, turns: "
                  f"{sum(len(c['turns']) for c in chains)}")
        return

    rows_path = out / "translations.jsonl"
    done = done_ids(rows_path)
    pending = 0
    if args.phase in ("main", "all"):
        pending += sum(j["job_id"] not in done for j in main_jobs(
            cfg, instances, models, parse_conditions(args.conditions)))
    if args.phase in ("s1", "all"):
        pending += sum(f"s1|{c['model']}|{c['strategy']}|{c['dialogue_id']}|t{t['turn_id']:03d}"
                       not in done for c in s1_chains(cfg, instances, dialogues, models)
                       for t in c["turns"])
    if not pending:
        print(f"[run] nothing left to run for phase {args.phase} in {out}", flush=True)
        return

    session = datetime.now().strftime("%Y%m%dT%H%M%S")
    write_artifacts(out, config_path, session, data_dir)
    ledger = CostLedger(out / "api_usage.jsonl", cfg["budget_usd"]["translation"])
    print(f"[run] {run_id} session {session} -> {out}; api spend so far ${ledger.spent():.4f} "
          f"of ${ledger.budget}", flush=True)
    gpu_snapshot(out, session, f"{args.phase}_start")
    backends = make_backends(cfg, models, ledger, out, session)
    gpu_snapshot(out, session, f"{args.phase}_loaded")
    runner = Runner(cfg, out, backends, ledger, latency_valid=not args.parallel, session=session)
    started = time.time()
    warmup(runner, cfg, models)
    if args.phase in ("main", "all"):
        run_main(runner, cfg, instances, dialogues, models, args)
    if args.phase in ("s1", "all"):
        run_s1(runner, cfg, instances, dialogues, models, args)
    gpu_snapshot(out, session, f"{args.phase}_end")
    print(f"[done] {args.phase} in {time.time() - started:.0f}s; api spend ${ledger.spent():.4f}",
          flush=True)


if __name__ == "__main__":
    main()
