"""Evaluation entry point.

Stage 1 (now):  python -m src.evaluate --mode teacher_pilot --teacher configs/teachers/gemma4_26b_a4b.yaml
Later:          --mode student  (added together with train_student.py)

Resumable: re-running skips prompts already in results/01_teacher_pilot/<teacher>.jsonl
"""
import argparse
import json
import statistics

from .metrics import quality_metrics
from .pilot_data import PROMPTS
from .teacher_loader import Teacher, load_config
from .utils import ROOT, read_jsonl


def teacher_pilot(args):
    cfg = load_config(args.teacher)
    name = cfg["name"]
    out_dir = ROOT / "results/01_teacher_pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.jsonl"
    done = {r["id"] for r in read_jsonl(path)} if path.exists() else set()
    teacher = Teacher(cfg, backend=args.backend)
    prompts = PROMPTS[: args.n]
    with open(path, "a", encoding="utf-8") as f:
        for i, p in enumerate(prompts):
            if i in done:
                continue
            out = teacher.generate(p)
            row = {"id": i, "teacher": name, "prompt": p, "response": out["text"],
                   "out_tokens": out["out_tokens"], "finish_reason": out["finish_reason"],
                   "tok_per_s": out["tok_per_s"], **quality_metrics(out["text"])}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(prompts)}] {row['out_tokens']} tok @ {row['tok_per_s']} tok/s "
                  f"geez={row['geez_ratio']} finish={row['finish_reason']}")
    rows = read_jsonl(path)
    print(f"\n== {name}: {len(rows)} responses ==")
    for k in ["geez_ratio", "latin_ratio", "cjk_chars", "distinct3", "out_tokens", "tok_per_s"]:
        print(f"{k:>14}: {statistics.mean(float(r[k]) for r in rows):.3f}")
    print(f"saved -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["teacher_pilot", "student"], default="teacher_pilot")
    ap.add_argument("--teacher", help="teacher YAML (teacher_pilot mode)")
    ap.add_argument("--backend", choices=["hf", "gguf"], help="override the YAML backend")
    ap.add_argument("--n", type=int, default=len(PROMPTS))
    args = ap.parse_args()
    if args.mode == "teacher_pilot":
        if not args.teacher:
            ap.error("--teacher is required for teacher_pilot")
        teacher_pilot(args)
    else:
        raise SystemExit("student mode is added with train_student.py (next stage)")


if __name__ == "__main__":
    main()
