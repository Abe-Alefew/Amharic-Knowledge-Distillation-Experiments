"""Score and compare completed teacher-generation JSONL files."""

import argparse
import json
import statistics
from pathlib import Path

from .metrics import quality_metrics
from .utils import ROOT, read_jsonl


def score_rows(rows):
    scored = []
    for row in rows:
        metrics = quality_metrics(row.get("response", ""))
        scored.append({**row, **metrics})
    if not scored:
        return {"n": 0}
    keys = ("geez_ratio", "latin_ratio", "distinct3", "n_words", "out_tokens", "tok_per_s")
    result = {"n": len(scored)}
    for key in keys:
        values = [float(row[key]) for row in scored if key in row]
        if values:
            result[f"mean_{key}"] = round(statistics.mean(values), 4)
    result["cjk_responses"] = sum(int(row["cjk_chars"]) > 0 for row in scored)
    result["truncated"] = sum(row.get("finish_reason") == "length" for row in scored)
    result["empty"] = sum(not row.get("response", "").strip() for row in scored)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="teacher output JSONL files")
    parser.add_argument("--output", default="results/02_generation/teacher_scores.json")
    args = parser.parse_args()

    report = {}
    for name in args.inputs:
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise SystemExit(f"input does not exist: {path}")
        report[path.stem] = score_rows(read_jsonl(path))
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved -> {output}")


if __name__ == "__main__":
    main()