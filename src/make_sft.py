"""Prepare teacher responses as a standard JSONL supervised fine-tuning set."""

import argparse

from .utils import ROOT, read_jsonl, write_jsonl


def make_records(rows, teacher=None, include_empty=False):
    """Convert generated rows to the chat structure consumed by train_student."""
    records = []
    for row in rows:
        prompt = str(row.get("prompt", "")).strip()
        response = str(row.get("response", "")).strip()
        if not prompt or (not response and not include_empty):
            continue
        if teacher and row.get("teacher") != teacher:
            continue
        records.append(
            {
                "id": row.get("id"),
                "task": row.get("task"),
                "source": row.get("source"),
                "teacher": row.get("teacher"),
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/teacher_outputs", help="JSONL file or directory")
    parser.add_argument("--output", default="data/sft/teacher_sft.jsonl")
    parser.add_argument("--teacher", help="include only one teacher name")
    parser.add_argument("--include-empty", action="store_true")
    args = parser.parse_args()

    input_path = ROOT / args.input
    paths = [input_path] if input_path.is_file() else sorted(input_path.glob("*.jsonl"))
    if not paths:
        raise SystemExit(f"no teacher JSONL files found under {input_path}")
    rows = [row for path in paths for row in read_jsonl(path)]
    records = make_records(rows, args.teacher, args.include_empty)
    if not records:
        raise SystemExit("no usable prompt/response pairs found")
    output = ROOT / args.output
    write_jsonl(output, records)
    print(f"wrote {len(records)} SFT records -> {output}")


if __name__ == "__main__":
    main()