"""Teacher x student tokenizer compatibility (required for logit distillation).

    python -m src.check_tokenizers --students google/gemma-4-E2B-it Qwen/Qwen3.5-2B

IDENTICAL -> logit KD is straightforward. PARTIAL/DIFFERENT -> needs cross-tokenizer
methods; fall back to response-level SFT. (Default student repo ids are guesses: verify.)
"""
import argparse
import csv

from .teacher_loader import list_teacher_configs, load_config
from .utils import ROOT

DEFAULT_STUDENTS = ["google/gemma-4-E2B-it", "google/gemma-4-E4B-it",
                    "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B"]


def compare(va, vb):
    if va == vb:
        return "IDENTICAL", 1.0, 1.0
    shared = set(va) & set(vb)
    overlap = len(shared) / max(len(va), len(vb))
    same_id = sum(va[t] == vb[t] for t in shared) / max(1, len(shared))
    return ("PARTIAL" if overlap > 0.9 and same_id > 0.99 else "DIFFERENT"), overlap, same_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*")
    ap.add_argument("--students", nargs="*", default=DEFAULT_STUDENTS)
    ap.add_argument("--out", default="results/01_teacher_pilot/tokenizer_match.csv")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    teachers = [load_config(p) for p in (args.configs or list_teacher_configs())]
    cache = {}

    def vocab(repo):
        if repo not in cache:
            try:
                cache[repo] = AutoTokenizer.from_pretrained(repo, trust_remote_code=True).get_vocab()
            except Exception as e:
                print(f"FAILED {repo}: {str(e)[:80]}")
                cache[repo] = None
        return cache[repo]

    rows = []
    for t in teachers:
        vt = vocab(t.get("tokenizer_repo") or t["hf_repo"])
        for s in args.students:
            vs = vocab(s)
            if vt is None or vs is None:
                continue
            verdict, ov, sid = compare(vt, vs)
            rows.append({"teacher": t["name"], "student": s, "teacher_vocab": len(vt),
                         "student_vocab": len(vs), "verdict": verdict,
                         "token_overlap": round(ov, 4), "same_id_rate": round(sid, 4)})
            print(f"{t['name']:<20} x {s:<28} {verdict:<10} overlap={ov:.3f} same_id={sid:.3f}")
    if rows:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
