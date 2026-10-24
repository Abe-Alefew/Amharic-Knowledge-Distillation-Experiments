"""Resumable teacher generation over the filtered prompt pool (no response filtering).

    python -m src.generate --teacher configs/teachers/gemma4_26b_a4b.yaml --limit 5      # smoke test
    python -m src.generate --teacher configs/teachers/gemma4_26b_a4b.yaml                # full run
    python -m src.generate --teacher ... --max-minutes 150                                # stop cleanly (Colab etc.)
    python -m src.generate --teacher ... --shard 0/2                                      # split across machines

Output: data/teacher_outputs/<teacher>.jsonl  (appended + flushed after every prompt)
        data/teacher_outputs/<teacher>.meta.json   (settings fingerprint; a resume with different settings is refused)
        data/teacher_outputs/<teacher>.errors.jsonl
        results/02_generation/<teacher>_summary.json
Quality metrics are stored as REPORT-ONLY columns; nothing is dropped here.
"""
import argparse
import hashlib
import json
import statistics
import time
from collections import defaultdict

from .metrics import quality_metrics
from .teacher_loader import Teacher, load_config
from .utils import ROOT, read_jsonl

FINGERPRINT_KEYS = ["hf_repo", "backend", "gguf", "hf", "chat_template_kwargs", "system_prompt",
                    "prompt_suffix", "strip_think", "generation"]


def file_sha(path):
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


def fmt_eta(seconds):
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h}h{m:02d}m"


def summarize(path):
    rows = read_jsonl(path)
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)

    def agg(rs):
        m = lambda k: statistics.mean(float(r[k]) for r in rs)
        return {"n": len(rs), "truncated_rate": round(sum(r["truncated"] for r in rs) / len(rs), 3),
                "empty": sum(r["empty"] for r in rs), "mean_out_tokens": round(m("out_tokens"), 1),
                "mean_tok_per_s": round(m("tok_per_s"), 2), "mean_geez_ratio": round(m("geez_ratio"), 3),
                "mean_latin_ratio": round(m("latin_ratio"), 3), "cjk_responses": sum(r["cjk_chars"] > 0 for r in rs),
                "mean_distinct3": round(m("distinct3"), 3)}
    return {"all": agg(rows), "by_task": {t: agg(rs) for t, rs in by_task.items()}} if rows else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="teacher YAML")
    ap.add_argument("--backend", choices=["hf", "gguf"], help="override the YAML backend")
    ap.add_argument("--prompts", default="data/prompts/prompts_filtered.jsonl")
    ap.add_argument("--out-dir", default="data/teacher_outputs")
    ap.add_argument("--limit", type=int, help="only the first N prompts (smoke test)")
    ap.add_argument("--task", help="only this task, e.g. summarization")
    ap.add_argument("--shard", help="i/n: process prompts[i::n]; output file gets a .shardIofN suffix")
    ap.add_argument("--max-new-tokens", type=int)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--max-minutes", type=float, help="stop cleanly after this long")
    ap.add_argument("--max-errors", type=int, default=5, help="abort after this many consecutive errors")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-raw", action="store_true", help="keep the raw output incl. <think> blocks")
    ap.add_argument("--force", action="store_true", help="resume even if settings differ from the first run")
    args = ap.parse_args()

    cfg = load_config(args.teacher)
    if args.max_new_tokens:
        cfg["generation"]["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        cfg["generation"]["temperature"] = args.temperature
    if args.backend:
        cfg["backend"] = args.backend
    name = cfg["name"]

    prompts_path = ROOT / args.prompts
    prompts = read_jsonl(prompts_path)
    if args.task:
        prompts = [p for p in prompts if p["task"] == args.task]
    suffix = ""
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        prompts = prompts[i::n]
        suffix = f".shard{i}of{n}"
    if args.limit:
        prompts = prompts[: args.limit]

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}{suffix}.jsonl"
    meta_path = out_dir / f"{name}{suffix}.meta.json"
    err_path = out_dir / f"{name}{suffix}.errors.jsonl"

    fingerprint = {k: cfg.get(k) for k in FINGERPRINT_KEYS}
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        diff = {k: (old["fingerprint"].get(k), fingerprint[k]) for k in FINGERPRINT_KEYS
                if old["fingerprint"].get(k) != fingerprint[k]}
        if diff and not args.force:
            raise SystemExit(f"Settings differ from the first run of {out.name}; refusing to mix them.\n"
                             f"{json.dumps(diff, ensure_ascii=False, indent=2)}\n"
                             f"Use --force to resume anyway, or write to a new --out-dir.")
        if old.get("prompts_sha") != file_sha(prompts_path):
            print("NOTE: the prompts file changed since the first run (ids already done are still skipped)")
    else:
        meta_path.write_text(json.dumps({"teacher": name, "fingerprint": fingerprint,
                                         "prompts_file": args.prompts, "prompts_sha": file_sha(prompts_path),
                                         "started": time.strftime("%Y-%m-%d %H:%M:%S")},
                                        ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [p for p in prompts if p["id"] not in done]
    print(f"{name}: {len(prompts)} prompts selected, {len(done)} already done, {len(todo)} to go")
    if not todo:
        print(json.dumps(summarize(out), ensure_ascii=False, indent=2))
        return

    teacher = Teacher(cfg)
    t_start, new_tokens, consecutive_errors, n_done = time.time(), 0, 0, 0
    try:
        with open(out, "a", encoding="utf-8") as f, open(err_path, "a", encoding="utf-8") as ef:
            for p in todo:
                if args.max_minutes and (time.time() - t_start) / 60 >= args.max_minutes:
                    print(f"--max-minutes reached; stopping cleanly after {n_done} prompts")
                    break
                try:
                    g = teacher.generate(p["prompt"])
                except Exception as e:
                    ef.write(json.dumps({"id": p["id"], "error": f"{type(e).__name__}: {str(e)[:300]}"},
                                        ensure_ascii=False) + "\n")
                    ef.flush()
                    consecutive_errors += 1
                    print(f"ERROR on {p['id']}: {type(e).__name__}: {str(e)[:120]}")
                    if consecutive_errors >= args.max_errors:
                        raise SystemExit(f"{consecutive_errors} consecutive errors; aborting (see {err_path.name})")
                    continue
                consecutive_errors = 0
                row = {"id": p["id"], "task": p["task"], "source": p["source"], "teacher": name,
                       "prompt": p["prompt"], "response": g["text"], "finish_reason": g["finish_reason"],
                       "truncated": g["finish_reason"] == "length", "empty": not g["text"].strip(),
                       "out_tokens": g["out_tokens"], "seconds": g["seconds"], "tok_per_s": g["tok_per_s"],
                       **quality_metrics(g["text"])}
                if args.save_raw and g["raw"] != g["text"]:
                    row["raw"] = g["raw"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                n_done += 1
                new_tokens += g["out_tokens"]
                if n_done % args.log_every == 0 or n_done == len(todo):
                    el = time.time() - t_start
                    eta = el / n_done * (len(todo) - n_done)
                    print(f"[{len(done) + n_done}/{len(prompts)}] {new_tokens / el:.1f} tok/s avg, "
                          f"{new_tokens / n_done:.0f} tok/resp, elapsed {fmt_eta(el)}, ETA {fmt_eta(eta)}")
    except KeyboardInterrupt:
        print("\ninterrupted; everything written so far is saved. Re-run the same command to resume.")

    summary = summarize(out)
    res = ROOT / "results/02_generation"
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{name}{suffix}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
    print(json.dumps(summary.get("all", {}), ensure_ascii=False, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()