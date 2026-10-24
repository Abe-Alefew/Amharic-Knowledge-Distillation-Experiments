"""Build the stage-1 pilot report.

    python -m src.report pilot            # blind judge sheet (shuffled A/B/C columns)
    python -m src.report pilot --no-blind

Outputs in results/01_teacher_pilot/: summary.md, judge_sheet.csv, judge_key.csv
"""
import argparse
import csv
import random
import statistics

from .utils import ROOT, read_jsonl


def pilot(args):
    d = ROOT / "results/01_teacher_pilot"
    data = {p.stem: read_jsonl(p) for p in sorted(d.glob("*.jsonl"))}
    if not data:
        raise SystemExit("no pilot results found; run src.evaluate first")
    lines = ["# Teacher pilot summary\n"]

    fert = d / "fertility.csv"
    if fert.exists():
        lines += ["## Tokenizer fertility (lower tok/word is better)\n",
                  "| name | tok/word | char/tok | parity vs EN | roundtrip | vocab |", "|---|---|---|---|---|---|"]
        for r in sorted(csv.DictReader(open(fert, encoding="utf-8")), key=lambda r: float(r["tok_per_word"])):
            lines.append(f"| {r['name']} | {r['tok_per_word']} | {r['chars_per_tok']} | {r['parity_vs_en']} "
                         f"| {r['roundtrip_ok']} | {r['vocab_size']} |")
        lines.append("")

    lines += ["## Auto-metrics on pilot generations (not a quality judgement)\n",
              "| teacher | n | geez | latin | cjk | distinct3 | truncated | out_tok | tok/s |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, rows in data.items():
        m = lambda k: statistics.mean(float(r[k]) for r in rows)
        trunc = sum(r["finish_reason"] == "length" for r in rows) / len(rows)
        lines.append(f"| {name} | {len(rows)} | {m('geez_ratio'):.2f} | {m('latin_ratio'):.2f} | "
                     f"{m('cjk_chars'):.1f} | {m('distinct3'):.2f} | {trunc:.2f} | {m('out_tokens'):.0f} | {m('tok_per_s'):.1f} |")
    (d / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    names = list(data)
    ids = sorted(set.intersection(*[{r["id"] for r in rows} for rows in data.values()]))
    by = {n: {r["id"]: r for r in rows} for n, rows in data.items()}
    rng = random.Random(0)
    letters = "ABCDEFGH"
    with open(d / "judge_sheet.csv", "w", newline="", encoding="utf-8-sig") as f, \
            open(d / "judge_key.csv", "w", newline="", encoding="utf-8") as k:
        w, kw = csv.writer(f), csv.writer(k)
        head = ["id", "prompt"]
        for i in range(len(names)):
            L = letters[i]
            head += [f"{L}_response", f"{L}_fluency_1to5", f"{L}_correct_1to5", f"{L}_natural_1to5"]
        w.writerow(head)
        kw.writerow(["id"] + [letters[i] for i in range(len(names))])
        for i in ids:
            order = names[:] if args.no_blind else rng.sample(names, len(names))
            w.writerow([i, by[names[0]][i]["prompt"]] + sum([[by[n][i]["response"], "", "", ""] for n in order], []))
            kw.writerow([i] + order)
    print((d / "summary.md").read_text(encoding="utf-8"))
    print(f"judge sheet -> {d / 'judge_sheet.csv'} (key: judge_key.csv)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["pilot"])
    ap.add_argument("--no-blind", action="store_true")
    args = ap.parse_args()
    pilot(args)


if __name__ == "__main__":
    main()
