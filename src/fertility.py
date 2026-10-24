"""Tokenizer fertility on REAL Amharic text (HornMT, human-translated news, CC BY 4.0).

    python -m src.fertility                       # all configs/teachers/*.yaml
    python -m src.fertility --extra-repos google/gemma-4-E2B-it Qwen/Qwen3.5-2B
    python -m src.fertility --source builtin      # no download

Metrics: tokens/word (lower = better), chars/token, parity vs English
(tokens of the Amharic line / tokens of the aligned English line), round-trip rate.
"""
import argparse
import csv
import re

from .teacher_loader import list_teacher_configs, load_config
from .utils import HORNMT_AM, HORNMT_EN, ROOT, download
from .pilot_data import AMHARIC_TEXT


def get_samples(source, n):
    if source == "hornmt":
        am = download(HORNMT_AM, ROOT / "data/raw/hornmt/amh.txt")
        en = download(HORNMT_EN, ROOT / "data/raw/hornmt/eng.txt")
        a = [l.strip() for l in open(am, encoding="utf-8")]
        e = [l.strip() for l in open(en, encoding="utf-8")]
        pairs = [(x, y) for x, y in zip(a, e) if x and y][:n]
        return [p[0] for p in pairs], [p[1] for p in pairs]
    return AMHARIC_TEXT, None


def _ws(s):
    return re.sub(r"\s+", "", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", help="teacher YAMLs (default: configs/teachers/*.yaml)")
    ap.add_argument("--extra-repos", nargs="*", default=[], help="extra HF repos, e.g. student models")
    ap.add_argument("--source", choices=["hornmt", "builtin"], default="hornmt")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default="results/01_teacher_pilot/fertility.csv")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    am, en = get_samples(args.source, args.n)
    words = sum(len(t.split()) for t in am)
    chars = sum(len(t) for t in am)
    print(f"{len(am)} Amharic lines ({args.source}): {words} words, {chars} chars\n")

    targets = []
    cfgs = [load_config(p) for p in (args.configs or list_teacher_configs())]
    for c in cfgs:
        targets.append((c["name"], c.get("tokenizer_repo") or c["hf_repo"]))
    for r in args.extra_repos:
        targets.append((r.split("/")[-1], r))

    rows = []
    for name, repo in targets:
        try:
            tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        except Exception as e:
            print(f"{name:<24} FAILED to load {repo}: {str(e)[:90]}")
            continue
        enc = [tok(t, add_special_tokens=False)["input_ids"] for t in am]
        n_tok = sum(len(x) for x in enc)
        rt = sum(_ws(tok.decode(x)) == _ws(t) for x, t in zip(enc, am)) / len(am)
        parity = ""
        if en:
            n_en = sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in en)
            parity = round(n_tok / n_en, 2)
        rows.append({"name": name, "repo": repo, "n_texts": len(am), "tokens": n_tok,
                     "tok_per_word": round(n_tok / words, 2), "chars_per_tok": round(chars / n_tok, 2),
                     "parity_vs_en": parity, "roundtrip_ok": round(rt, 3), "vocab_size": len(tok)})

    print(f"{'name':<24}{'tok/word':>9}{'char/tok':>9}{'parity':>8}{'roundtrip':>10}{'vocab':>9}")
    for r in sorted(rows, key=lambda r: r["tok_per_word"]):
        print(f"{r['name']:<24}{r['tok_per_word']:>9}{r['chars_per_tok']:>9}{str(r['parity_vs_en']):>8}"
              f"{r['roundtrip_ok']:>10}{r['vocab_size']:>9}")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
