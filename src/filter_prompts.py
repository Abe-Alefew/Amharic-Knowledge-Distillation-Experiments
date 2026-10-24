"""Filter the raw prompt pool (stage after build_prompts, before teacher generation).

    python -m src.filter_prompts
    python -m src.filter_prompts --tokenizer-repo google/gemma-4-26B-A4B-it   # exact token counts

Order: license/provenance -> length -> token budget -> script ratio -> PII/links -> noise
       -> exact dedupe -> near-dup dedupe (MinHash+LSH) -> overlap with data/evaluation.jsonl
Outputs: prompts_filtered.jsonl, prompts_rejected.jsonl (with reasons), filter_report.json
"""
import argparse
import json
import re
import unicodedata
import zlib
from collections import Counter, defaultdict

import numpy as np

from .metrics import geez_ratio
from .utils import ROOT, load_yaml, normalize_text, read_jsonl, sha, write_jsonl

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?:\+?251|\b0)[\s-]?[79]\d[\s-]?\d{3}[\s-]?\d{4}|\b\d{9,}\b")
LINK = re.compile(r"https?://|www\.|t\.me/|(?<!\w)@\w{3,}")
EVAL_KEYS = ("prompt", "instruction", "question", "input", "text")
P = 2147483647  # 2^31 - 1


def wc(s):
    return len(s.split())


# ------------------------------------------------------------------ rule checks
def noise_ratio(s):
    bad = 0
    for c in s:
        if c.isspace() or c.isalnum() or unicodedata.category(c).startswith("P"):
            continue
        bad += 1
    return bad / max(1, len(s))


def reject_reason(r, cfg, count_tokens):
    t, p = r["task"], r["prompt"]
    if r["license"] not in cfg["license_allowlist"]:
        return "license_not_allowed"
    if r["provenance"] not in cfg["allowed_provenance"]:
        return "provenance_not_allowed"
    n = wc(p)
    lo, hi = cfg["word_bounds"].get(t, cfg["word_bounds"]["default"])
    if n < lo:
        return "too_short"
    if n > hi:
        return "too_long"
    if count_tokens(p) > cfg["max_prompt_tokens"]:
        return "too_many_tokens"
    if geez_ratio(p) < cfg["min_geez_ratio"].get(t, cfg["min_geez_ratio"]["default"]):
        return "low_geez_ratio"
    if EMAIL.search(p) or PHONE.search(p) or LINK.search(p):
        return "pii_or_link"
    nz = cfg["noise"]
    if noise_ratio(p) > nz["max_symbol_ratio"]:
        return "symbol_noise"
    if re.search(r"(.)\1{%d,}" % nz["max_char_run"], p):
        return "char_repeat"
    words = p.split()
    if len(words) >= 8 and len(set(words)) / len(words) < nz["min_distinct_word_ratio"]:
        return "repetitive"
    if any(b in p for b in nz["blocklist_substrings"]):
        return "blocklisted"
    return None


# ------------------------------------------------------------------ dedupe
def shingles(text, n=5):
    s = re.sub(r"\s+", " ", text.strip())
    return {zlib.crc32(s[i:i + n].encode("utf-8")) % P for i in range(max(1, len(s) - n + 1))}


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def near_dedupe(records, sets, thr, bands=16, rows=4):
    rng = np.random.RandomState(0)
    a = rng.randint(1, P, size=bands * rows).astype(np.uint64)
    b = rng.randint(0, P, size=bands * rows).astype(np.uint64)
    buckets = [defaultdict(list) for _ in range(bands)]
    keep, dropped = [], []
    for i, r in enumerate(records):
        arr = np.fromiter(sets[i], dtype=np.uint64)
        sig = ((arr[None, :] * a[:, None] + b[:, None]) % P).min(axis=1)
        keys = [sig[k * rows:(k + 1) * rows].tobytes() for k in range(bands)]
        cand = set()
        for k in range(bands):
            cand.update(buckets[k].get(keys[k], ()))
        dup = next((j for j in cand if jaccard(sets[i], sets[j]) >= thr), None)
        if dup is None:
            keep.append(i)
            for k in range(bands):
                buckets[k][keys[k]].append(i)
        else:
            dropped.append((i, records[dup]["id"]))
    return keep, dropped


def read_eval_texts(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    texts = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            texts.append(line)
            continue
        if isinstance(obj, str):
            texts.append(obj)
        elif isinstance(obj, dict):
            key = next((k for k in EVAL_KEYS if isinstance(obj.get(k), str)), None)
            texts += [obj[key]] if key else [v for v in obj.values() if isinstance(v, str) and len(v) > 8]
    return [normalize_text(t) for t in texts if t.strip()]


def eval_overlap(records, eval_texts, d):
    esets = [shingles(t) for t in eval_texts]
    flagged = {}
    for i, r in enumerate(records):
        full = shingles(r["prompt"])
        content_text = r.get("meta", {}).get("source_text")
        content = shingles(content_text) if content_text else full
        for k, e in enumerate(esets):
            if jaccard(full, e) >= d["eval_jaccard"]:
                flagged[i] = k
                break
            m = min(len(content), len(e))
            if m >= d["min_containment_shingles"] and len(content & e) / m >= d["eval_containment"]:
                flagged[i] = k
                break
    return flagged


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prompt_filter.yaml")
    ap.add_argument("--tokenizer-repo", help="exact token counts instead of the words*k estimate")
    args = ap.parse_args()
    cfg = load_yaml(ROOT / args.config)

    if args.tokenizer_repo:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer_repo, trust_remote_code=True)
        count_tokens = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    else:
        count_tokens = lambda s: int(wc(s) * cfg["est_tokens_per_word"])

    records = read_jsonl(ROOT / cfg["in"])
    rejected, passed = [], []
    for r in records:
        why = reject_reason(r, cfg, count_tokens)
        (rejected.append({**r, "reject_reason": why}) if why else passed.append(r))

    # exact duplicates on normalised text
    seen, unique = {}, []
    for r in passed:
        h = sha(re.sub(r"[\s\W_]+", "", r["prompt"].lower()))
        if h in seen:
            rejected.append({**r, "reject_reason": f"exact_duplicate_of:{seen[h]}"})
        else:
            seen[h] = r["id"]
            unique.append(r)

    # near duplicates (MinHash + LSH, verified with exact Jaccard)
    d = cfg["dedupe"]
    sets = [shingles(r["prompt"]) for r in unique]
    keep, dropped = near_dedupe(unique, sets, d["near_dup_jaccard"])
    for i, other in dropped:
        rejected.append({**unique[i], "reject_reason": f"near_duplicate_of:{other}"})
    unique = [unique[i] for i in keep]

    # overlap with the evaluation prompts
    eval_path = ROOT / cfg["evaluation_file"]
    eval_texts = read_eval_texts(eval_path)
    if not eval_texts:
        print(f"WARNING: {eval_path} is missing or empty -> evaluation-overlap check SKIPPED")
    flagged = eval_overlap(unique, eval_texts, d) if eval_texts else {}
    for i, k in flagged.items():
        rejected.append({**unique[i], "reject_reason": f"overlaps_evaluation_prompt:{k}"})
    final = [r for i, r in enumerate(unique) if i not in flagged]
    for r in final:
        r["n_words"] = wc(r["prompt"])
        r["est_tokens"] = count_tokens(r["prompt"])

    write_jsonl(ROOT / cfg["out"], final)
    write_jsonl(ROOT / cfg["rejected"], [{**x, "prompt": x["prompt"][:300]} for x in rejected])

    def by(rows, f):
        return dict(Counter(f(x) for x in rows))
    report = {
        "input": len(records), "kept": len(final), "rejected": len(rejected),
        "evaluation_prompts_checked": len(eval_texts),
        "kept_by_task": by(final, lambda x: x["task"]), "kept_by_source": by(final, lambda x: x["source"]),
        "raw_by_task": by(records, lambda x: x["task"]),
        "reject_reasons": by(rejected, lambda x: x["reject_reason"].split(":")[0]),
        "reject_by_source": by(rejected, lambda x: x["source"]),
        "est_tokens_mean": round(float(np.mean([x["est_tokens"] for x in final])), 1) if final else 0,
        "est_tokens_max": max((x["est_tokens"] for x in final), default=0),
        "token_counting": args.tokenizer_repo or "estimate",
    }
    (ROOT / cfg["report"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
