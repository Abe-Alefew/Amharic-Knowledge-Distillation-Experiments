"""Build the evaluation prompt set (60-100 prompts) from real, human-made Amharic datasets.

    python -m src.build_eval --inspect          # print columns + first row of every enabled dataset (check schemas!)
    python -m src.build_eval                    # -> data/evaluation.jsonl + data/evaluation_report.json
    python -m src.build_eval --enable ethionlp_instructions

Record: id, task, source, license, provenance, prompt, reference, answer_type, meta
answer_type: mcq_1to4 | mcq_letter | translation | free_text | weak_reference
Run this BEFORE filter_prompts: the filter then removes held-out items (Aya, news, HornMT pairs) from
the training pool, so the evaluation prompts never leak into training.
"""
import argparse
import ast
import json
import random
from collections import Counter, defaultdict

from .build_prompts import URL_RE, wc
from .metrics import geez_ratio
from .utils import HORNMT_AM, HORNMT_EN, ROOT, download, load_yaml, normalize_text, write_jsonl


def _load(c):
    from datasets import load_dataset
    return load_dataset(c["hf_repo"], c.get("hf_config"), split=c.get("split", "train"))


def _first(row, names):
    for k in names:
        if isinstance(row.get(k), str) and row[k].strip():
            return k
    return None


def _round_robin(groups, rng, n):
    for g in groups.values():
        rng.shuffle(g)
    out, keys, i = [], sorted(groups), 0
    while len(out) < n and any(groups[k] for k in keys):
        k = keys[i % len(keys)]
        i += 1
        if groups[k]:
            out.append(groups[k].pop())
    return out


def _rec(c, prompt, reference, answer_type, meta):
    return {"task": c["task"], "license": c["license"], "provenance": c["provenance"], "prompt": prompt,
            "reference": reference, "answer_type": answer_type, "meta": meta}


# ------------------------------------------------------------------ sources
def src_belebele(c, rng, T, P):
    ds = _load(c)
    seen, rows = set(), []
    for r in ds:
        passage = normalize_text(r["flores_passage"])
        if passage in seen:                      # one question per passage -> more diverse items
            continue
        seen.add(passage)
        rows.append(r)
    rng.shuffle(rows)
    out = []
    for r in rows[: c["n"]]:
        prompt = (T["mcq_rc"].replace("{passage}", normalize_text(r["flores_passage"]))
                  .replace("{question}", normalize_text(r["question"])))
        for i in range(1, 5):
            prompt = prompt.replace("{a%d}" % i, normalize_text(r[f"mc_answer{i}"]))
        out.append(_rec(c, prompt, str(r["correct_answer_num"]), "mcq_1to4",
                        {"question_number": r.get("question_number"), "link": r.get("link")}))
    return out


def _choices(v):
    if isinstance(v, (list, tuple)):
        return list(v)
    try:
        return list(ast.literal_eval(v))
    except Exception:
        return list(json.loads(v))


def src_afrimmlu(c, rng, T, P):
    ds = _load(c)
    groups = defaultdict(list)
    for r in ds:
        ch = _choices(r["choices"])
        if len(ch) == 4:
            groups[r["subject"]].append((r, ch))
    out = []
    for r, ch in _round_robin(groups, rng, c["n"]):
        prompt = T["mcq_knowledge"].replace("{question}", normalize_text(r["question"]))
        for i in range(4):
            prompt = prompt.replace("{a%d}" % (i + 1), normalize_text(str(ch[i])))
        ans = r["answer"]
        ans = "ABCD"[ans] if isinstance(ans, int) else str(ans).strip().upper()
        out.append(_rec(c, prompt, ans, "mcq_letter", {"subject": r["subject"]}))
    return out


def src_hornmt(c, rng, T, P):
    a = [normalize_text(l) for l in open(download(HORNMT_AM, ROOT / "data/raw/hornmt/amh.txt"), encoding="utf-8")]
    e = [normalize_text(l) for l in open(download(HORNMT_EN, ROOT / "data/raw/hornmt/eng.txt"), encoding="utf-8")]
    ok = [i for i, (x, y) in enumerate(zip(a, e))
          if x and y and c["min_words"] <= wc(x) <= c["max_words"] and geez_ratio(x) >= 0.9]
    rng.shuffle(ok)
    out = []
    for k, i in enumerate(ok[: c["n"]]):
        d = "am2en" if k % 2 == 0 else "en2am"
        src, ref = (a[i], e[i]) if d == "am2en" else (e[i], a[i])
        prompt = rng.choice(P["translation_" + d]).replace("{text}", src)
        out.append(_rec(c, prompt, ref, "translation",
                        {"direction": d, "pair_index": i, "source_text": src}))
    return out


def src_aya(c, rng, T, P):
    ds = _load({**c, "split": "train"})
    idx = [i for i, l in enumerate(ds["language"]) if l == "Amharic"]
    rows = []
    for r in ds.select(idx):
        p = normalize_text(r["inputs"])
        if (c["min_words"] <= wc(p) <= c["max_words"] and geez_ratio(p) >= c["min_geez_ratio"]
                and not URL_RE.search(p) and normalize_text(r["targets"])):
            rows.append((p, normalize_text(r["targets"])))
    rng.shuffle(rows)
    return [_rec(c, p, t, "free_text", {}) for p, t in rows[: c["n"]]]


def src_news(c, rng, T, P):
    ds = _load({**c, "split": "train"})
    groups = defaultdict(list)
    for r in ds:
        art = normalize_text(r["article"])
        if (c["min_words"] <= wc(art) <= c["max_words"] and geez_ratio(art) >= c["min_geez_ratio"]
                and not URL_RE.search(art)):
            groups[r["category"]].append((art, r))
    out = []
    for art, r in _round_robin(groups, rng, c["n"]):
        prompt = rng.choice(P["summarization"]).replace("{text}", art)
        out.append(_rec(c, prompt, normalize_text(r["headline"]), "weak_reference",
                        {"category": r["category"], "source_text": art,
                         "note": "reference is the headline, not a summary"}))
    return out


def src_ethionlp(c, rng, T, P):
    ds = _load(c)
    rows = []
    for r in ds:
        ik = _first(r, ["instruction", "prompt", "question"])
        ok = _first(r, ["output", "response", "answer", "target"])
        if not ik or not ok:
            continue
        inp = r.get("input") if isinstance(r.get("input"), str) else ""
        p = normalize_text(r[ik] + ("\n\n" + inp if inp.strip() else ""))
        if c["min_words"] <= wc(p) <= c["max_words"] and geez_ratio(p) >= c["min_geez_ratio"] and not URL_RE.search(p):
            rows.append((p, normalize_text(r[ok]), r.get("source")))
    rng.shuffle(rows)
    return [_rec(c, p, t, "free_text", {"orig_source": s}) for p, t, s in rows[: c["n"]]]


LOADERS = {"belebele": src_belebele, "afrimmlu": src_afrimmlu, "hornmt": src_hornmt, "aya": src_aya,
           "news": src_news, "ethionlp_instructions": src_ethionlp}


def inspect(cfg):
    for name, c in cfg["sources"].items():
        if not c.get("enabled") or name == "hornmt":
            continue
        try:
            ds = _load({**c, "split": c.get("split", "train")})
            row = ds[0]
            print(f"\n== {name} ({c['hf_repo']}) columns={list(row)}")
            print({k: (str(v)[:80] if v is not None else None) for k, v in row.items()})
        except Exception as e:
            print(f"\n== {name}: FAILED to load ({type(e).__name__}: {str(e)[:150]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval_prompts.yaml")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--enable", nargs="*", default=[])
    ap.add_argument("--disable", nargs="*", default=[])
    ap.add_argument("--out")
    args = ap.parse_args()
    cfg = load_yaml(ROOT / args.config)
    for n in args.enable:
        cfg["sources"][n]["enabled"] = True
    for n in args.disable:
        cfg["sources"][n]["enabled"] = False
    if args.inspect:
        return inspect(cfg)

    rng = random.Random(cfg["seed"])
    P = load_yaml(ROOT / "configs/prompts.yaml")["templates"]
    T = cfg["templates"]
    final, warnings = [], []
    for name, c in cfg["sources"].items():
        if not c.get("enabled"):
            continue
        recs = LOADERS[name](c, rng, T, P)
        print(f"{name}: {len(recs)}/{c['n']} items")
        if len(recs) < c["n"]:
            warnings.append(f"{name}: only {len(recs)} of {c['n']} requested")
        if c["license"] in ("unspecified", "unverified"):
            warnings.append(f"{name}: license is '{c['license']}'")
        for r in recs:
            r["source"] = name
        final += recs
    rng.shuffle(final)
    counters = defaultdict(int)
    for r in final:
        counters[r["task"]] += 1
        r["id"] = f"eval-{r['task']}-{counters[r['task']]:03d}"
    final = [{k: r[k] for k in ("id", "task", "source", "license", "provenance", "prompt", "reference",
                                "answer_type", "meta")} for r in final]
    if not (cfg["min_total"] <= len(final) <= cfg["max_total"]):
        warnings.append(f"total {len(final)} is outside {cfg['min_total']}-{cfg['max_total']}")
    out = ROOT / (args.out or cfg["out"])
    write_jsonl(out, final)
    report = {"total": len(final), "by_task": dict(Counter(r["task"] for r in final)),
              "by_source": dict(Counter(r["source"] for r in final)),
              "by_answer_type": dict(Counter(r["answer_type"] for r in final)),
              "words_mean": round(sum(wc(r["prompt"]) for r in final) / max(1, len(final)), 1),
              "warnings": warnings}
    (ROOT / cfg["report"]).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {len(final)} evaluation prompts -> {out}")


if __name__ == "__main__":
    main()
