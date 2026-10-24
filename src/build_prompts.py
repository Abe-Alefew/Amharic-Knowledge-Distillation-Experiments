"""Build the raw prompt pool from REAL Amharic datasets (stage: prompt design).

    python -m src.build_prompts                          # configs/prompts.yaml -> data/prompts/prompts_raw.jsonl
    python -m src.build_prompts --total 300 --only hornmt news     # quick debug build
    python -m src.build_prompts --disable wikipedia      # drop a source

Each record: id, task, source, license, provenance, prompt, meta.
Dedupe vs data/evaluation.jsonl, license allowlist and quality thresholds are done in the
NEXT stage (filter_prompts.py); this stage only extracts, normalises and samples.
"""
import argparse
import json
import random
import re
import statistics
from collections import defaultdict

from .metrics import geez_ratio
from .utils import HORNMT_AM, HORNMT_EN, ROOT, download, load_yaml, normalize_text, sha, write_jsonl

URL_RE = re.compile(r"https?://|www\.")
SOCIAL_RE = re.compile(r"@\w|#\w|t\.me/")
Q_END = ("?", "፧")


def wc(s):
    return len(s.split())


def _render(rng, T, key, text):
    return rng.choice(T[key]).replace("{text}", text)


def _detect_col(row):
    for k in ("text", "sentence", "sentences", "content"):
        if k in row and isinstance(row[k], str):
            return k
    for k, v in row.items():
        if isinstance(v, str):
            return k
    raise KeyError("no string column found; set text_column in the config")


def _is_question(s, c):
    if not s.endswith(Q_END) or (s.count("?") + s.count("፧")) != 1:
        return False
    if not (c["min_words"] <= wc(s) <= c["max_words"]):
        return False
    if URL_RE.search(s) or SOCIAL_RE.search(s):
        return False
    if sum(ch.isdigit() for ch in s) / len(s) > 0.15:
        return False
    return geez_ratio(s) >= c["min_geez_ratio"]


# ---------------------------------------------------------------- loaders
# Each loader returns a shuffled list of {"prompt": str, "meta": dict}.

def load_aya(c, rng, limit, T):
    from datasets import load_dataset
    ds = load_dataset(c["hf_repo"], split="train")
    idx = [i for i, l in enumerate(ds["language"]) if l == "Amharic"]
    out, seen = [], set()
    for r in ds.select(idx):
        p = normalize_text(r["inputs"])
        if len(p) < c["min_chars"] or wc(p) > c["max_words"] or geez_ratio(p) < c["min_geez_ratio"]:
            continue
        if URL_RE.search(p) or sha(p) in seen:
            continue
        seen.add(sha(p))
        meta = {"annotation_type": r.get("annotation_type")}
        if c.get("store_references"):
            meta["reference"] = normalize_text(r["targets"])
        out.append({"prompt": p, "meta": meta})
    print(f"  aya: {len(idx)} Amharic rows, {len(out)} usable")
    rng.shuffle(out)
    return out[:limit]


def load_sentence_questions(c, rng, limit, T):
    from datasets import load_dataset
    ds = load_dataset(c["hf_repo"], split="train", streaming=True)
    ds = ds.shuffle(seed=rng.randrange(10 ** 6), buffer_size=c["shuffle_buffer"])
    col, out, seen, scanned = c.get("text_column"), [], set(), 0
    for r in ds:
        scanned += 1
        if col is None:
            col = _detect_col(r)
        s = normalize_text(str(r[col]))
        if _is_question(s, c) and sha(s) not in seen:
            seen.add(sha(s))
            out.append({"prompt": s, "meta": {}})
        if len(out) >= limit or scanned >= c["max_scan"]:
            break
    print(f"  sentence_questions: scanned {scanned} rows (column '{col}'), kept {len(out)} questions")
    return out


def load_news(c, rng, limit, T):
    from datasets import load_dataset
    ds = load_dataset(c["hf_repo"], split="train")
    by_cat = defaultdict(list)
    for r in ds:
        art = normalize_text(r["article"])
        if not (c["min_words"] <= wc(art) <= c["max_words"]):
            continue
        if URL_RE.search(art) or geez_ratio(art) < c["min_geez_ratio"]:
            continue
        by_cat[r["category"]].append((art, r))
    for v in by_cat.values():
        rng.shuffle(v)
    out, seen = [], set()
    cats = sorted(by_cat)
    i = 0
    while len(out) < limit and any(by_cat[k] for k in cats):   # round-robin over categories
        k = cats[i % len(cats)]
        i += 1
        if not by_cat[k]:
            continue
        art, r = by_cat[k].pop()
        if sha(art) in seen:
            continue
        seen.add(sha(art))
        out.append({"prompt": _render(rng, T, "summarization", art),
                    "meta": {"category": k, "headline": normalize_text(r["headline"]),
                             "article_words": wc(art), "source_text": art}})
    print(f"  news: categories={ {k: len(v) for k, v in by_cat.items()} } (left after sampling), kept {len(out)}")
    return out


def load_hornmt(c, rng, limit, T):
    am = download(HORNMT_AM, ROOT / "data/raw/hornmt/amh.txt")
    en = download(HORNMT_EN, ROOT / "data/raw/hornmt/eng.txt")
    a = [normalize_text(l) for l in open(am, encoding="utf-8")]
    e = [normalize_text(l) for l in open(en, encoding="utf-8")]
    if len(a) != len(e):
        raise RuntimeError(f"HornMT line mismatch: amh={len(a)} eng={len(e)}")
    out = []
    for idx, (x, y) in enumerate(zip(a, e)):
        if not x or not y or not (c["min_words"] <= wc(x) <= c["max_words"]) or geez_ratio(x) < c["min_geez_ratio"]:
            continue
        dirs = ["am2en", "en2am"] if c["both_directions"] else [rng.choice(["am2en", "en2am"])]
        for d in dirs:
            src, ref = (x, y) if d == "am2en" else (y, x)
            out.append({"prompt": _render(rng, T, f"translation_{d}", src),
                        "meta": {"direction": d, "reference": ref, "pair_index": idx,
                                 "source_text": src}})
    print(f"  hornmt: {len(a)} aligned pairs, {len(out)} prompts")
    rng.shuffle(out)
    return out[:limit]


def load_wikipedia(c, rng, limit, T):
    from datasets import load_dataset
    ds = load_dataset(c["hf_repo"], c["hf_config"], split="train")
    out, seen, stats = [], set(), defaultdict(int)
    for r in ds:
        title = normalize_text(r["title"])
        text = r["text"] or ""
        if not title or any(x in title for x in c["exclude_title_substrings"]):
            stats["excluded_title"] += 1
        elif wc(title) > c["max_title_words"] or geez_ratio(title) < c["min_title_geez_ratio"]:
            stats["bad_title"] += 1
        elif re.search(r"[0-9\u1369-\u137C]", title):
            stats["digit_title"] += 1          # years, dates, numbered pages
        elif len(text) < c["min_text_chars"]:
            stats["stub"] += 1
        elif sha(title) in seen:
            stats["dup_title"] += 1
        else:
            seen.add(sha(title))
            out.append({"prompt": _render(rng, T, "wiki_knowledge", title),
                        "meta": {"source_text": title, "url": r["url"],
                                 "lead": normalize_text(text)[:400]}})
    print(f"  wikipedia: {len(ds)} articles, usable titles={len(out)}, rejected={dict(stats)}")
    rng.shuffle(out)
    return out[:limit]


LOADERS = {"aya": load_aya, "sentence_questions": load_sentence_questions, "news": load_news,
           "hornmt": load_hornmt, "wikipedia": load_wikipedia}


# ---------------------------------------------------------------- planning / selection
def plan_quotas(cfg):
    tasks = {}
    for t, tc in cfg["tasks"].items():
        srcs = {s: w for s, w in tc["sources"].items() if cfg["sources"][s].get("enabled", True)}
        if srcs:
            tasks[t] = (tc["weight"], srcs)
    total_w = sum(w for w, _ in tasks.values())
    quotas = {}
    for t, (w, srcs) in tasks.items():
        tt = round(cfg["target_total"] * w / total_w)
        sw = sum(srcs.values())
        for s, sv in srcs.items():
            quotas[(t, s)] = round(tt * sv / sw)
    return quotas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prompts.yaml")
    ap.add_argument("--out")
    ap.add_argument("--total", type=int)
    ap.add_argument("--only", nargs="*", help="build from these sources only")
    ap.add_argument("--enable", nargs="*", default=[])
    ap.add_argument("--disable", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_yaml(ROOT / args.config)
    for n in args.enable:
        cfg["sources"][n]["enabled"] = True
    for n in args.disable:
        cfg["sources"][n]["enabled"] = False
    if args.only:
        for n in cfg["sources"]:
            cfg["sources"][n]["enabled"] = n in args.only
    if args.total:
        cfg["target_total"] = args.total
    rng = random.Random(cfg["seed"])
    T = cfg["templates"]
    quotas = plan_quotas(cfg)

    warnings, detail, left_by_task = [], {}, defaultdict(list)
    selected, short_by_task = [], defaultdict(int)
    for (t, s), q in quotas.items():
        c = cfg["sources"][s]
        print(f"[{t}/{s}] quota={q}")
        if c.get("provenance") != "human":
            warnings.append(f"source '{s}' has provenance '{c.get('provenance')}' (not human-written)")
        cands = LOADERS[s](c, rng, int(q * cfg["candidate_factor"]) + 1, T)
        for x in cands:
            x.update(task=t, source=s, license=c["license"], provenance=c["provenance"])
        take, left = cands[:q], cands[q:]
        selected += take
        left_by_task[t] += left
        short_by_task[t] += q - len(take)
        detail[f"{t}/{s}"] = {"quota": q, "loaded": len(cands), "taken": len(take)}

    for t, short in short_by_task.items():           # refill shortfalls within the same task
        if short > 0:
            pool = left_by_task[t]
            rng.shuffle(pool)
            fill = pool[:short]
            selected += fill
            detail[f"{t}/_refill"] = {"short": short, "filled": len(fill)}
            if len(fill) < short:
                warnings.append(f"task '{t}' is {short - len(fill)} prompts short of its quota")

    seen, final = set(), []
    for x in selected:                                 # exact-dup removal across the whole pool
        h = sha(x["prompt"])
        if h not in seen:
            seen.add(h)
            final.append(x)
    rng.shuffle(final)
    counters = defaultdict(int)
    for x in final:
        counters[(x["task"], x["source"])] += 1
        x["id"] = f"{x['task']}-{x['source']}-{counters[(x['task'], x['source'])]:05d}"
    final = [{k: x[k] for k in ("id", "task", "source", "license", "provenance", "prompt", "meta")}
             for x in final]

    out = ROOT / (args.out or cfg["out"])
    write_jsonl(out, final)
    lens = [wc(x["prompt"]) for x in final]
    report = {"total": len(final), "by_task": {}, "by_source": {}, "by_provenance": {},
              "selection_detail": detail, "prompt_words_mean": round(statistics.mean(lens), 1),
              "prompt_words_median": statistics.median(lens), "prompt_words_max": max(lens),
              "warnings": warnings}
    for x in final:
        for key, fld in (("by_task", "task"), ("by_source", "source"), ("by_provenance", "provenance")):
            report[key][x[fld]] = report[key].get(x[fld], 0) + 1
    (out.parent / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    print(f"\nwrote {len(final)} prompts -> {out}")
    print(json.dumps({k: report[k] for k in ("by_task", "by_source", "by_provenance", "warnings")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
