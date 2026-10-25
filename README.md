# Amharic capability distillation

This repository experiments with distilling Amharic capabilities from a
larger teacher model into a smaller student model:

```text
candidate teacher selection
        -> evaluation-set construction
        -> training-prompt construction and filtering
        -> teacher response generation
        -> SFT dataset preparation
        -> student fine-tuning
        -> held-out evaluation
```

## Current status

The data-preparation stages have produced a complete 4,500-prompt training
pool and a 90-example held-out evaluation set. The `results/` directory
currently contains tokenizer-fertility measurements only; teacher-generation
and student-training output directories are still empty apart from `.gitkeep`.

## 1. Candidate teacher tokenizer-fertility test

The fertility test used 500 real Amharic lines from HornMT and compared the
configured teacher tokenizers. Lower `tok_per_word` and higher `chars_per_tok`
usually indicate more efficient Amharic tokenization. All three candidates
round-tripped successfully on this check.

| Candidate | Repository | Tokens | Tokens/word | Characters/token | Parity vs English | Round-trip |
|---|---|---:|---:|---:|---:|---:|
| `gemma4_26b_a4b` | `google/gemma-4-26B-A4B-it` | 29,085 | 3.33 | 1.53 | 2.25 | 1.000 |
| `qwen3_5_27b` | `Qwen/Qwen3.5-27B` | 61,447 | 7.03 | 0.72 | 4.70 | 1.000 |
| `qwen3_6_35b_a3b` | `Qwen/Qwen3.6-35B-A3B` | 61,447 | 7.03 | 0.72 | 4.70 | 1.000 |

On tokenizer efficiency alone, `gemma4_26b_a4b` is the strongest candidate:
it uses fewer than half as many tokens per Amharic word as either Qwen
candidate. This is a tokenizer result, not a quality or generation-speed
benchmark; response quality still needs to be measured after generation.

The saved result is [`results/01_teacher_pilot/fertility.csv`](<./results/01_teacher_pilot/fertility.csv>).

Command used:

```bash
python -m src.fertility
```

The command uses the teacher YAML files under
[`configs/teachers/`](<./configs/teachers/>) and writes
`results/01_teacher_pilot/fertility.csv`. To test additional tokenizers:

```bash
python -m src.fertility --extra-repos google/gemma-4-E2B-it Qwen/Qwen3.5-2B
```

## 2. Evaluation set

The held-out evaluation set contains 90 human-provenance examples:

| Task | Count | Source | Answer type |
|---|---:|---|---|
| Translation | 20 | HornMT | Translation reference |
| Instruction | 20 | Aya | Free text |
| Knowledge MCQ | 20 | AfriMMLU | A-D letter |
| Reading comprehension | 20 | Belebele | 1-4 option |
| Summarization | 10 | Amharic news | Weak reference/headline |

The aggregate report shows a mean prompt length of 69.9 words. One metadata
warning remains: AfriMMLU has an `unspecified` license and should be checked
before redistribution.

Example from [`data/evaluation.jsonl`](<./data/evaluation.jsonl>) (the
reference is for scoring only and must not be sent to the model):

```json
{"id":"eval-instruction-001","task":"instruction","source":"aya",
 "prompt":"ጥያቄ፦ ኮርቬት የተዋወቀችው በየትኛው ዓመት ነው? መልስ፦",
 "reference":"1953","answer_type":"free_text"}
```

The full summary is in
[`data/evaluation_report.json`](<./data/evaluation_report.json>). Rebuild it
with:

```bash
python -m src.build_eval
```

This file intentionally contains answers/references because the evaluator
needs them. Generation code must consume only `prompt`; `reference` is a
scoring label and must remain held out from training and model input.

## 3. Prompt building

The prompt builder selected 4,500 human-provenance prompts:

| Task | Count |
|---|---:|
| QA | 1,800 |
| Translation | 900 |
| Summarization | 1,125 |
| Wikipedia knowledge | 675 |

Source totals were 1,170 sentence questions, 900 HornMT translations, 1,125
news prompts, 675 Wikipedia prompts, and 630 Aya prompts. The mean prompt
length was 71.6 words, the median was 17, and the maximum was 511 words.
No build warnings were reported.

Example from
[`data/prompts/prompts_raw.jsonl`](<./data/prompts/prompts_raw.jsonl>):

```json
{"id":"qa-sentence_questions-00001","task":"qa",
 "source":"sentence_questions",
 "prompt":"፨፨ ኮምፒዩተራችን እንዳለው እና እንደሌለው እንዴት ማወቅ እንችላለን?"}
```

The raw-build summary is
[`data/prompts/build_report.json`](<./data/prompts/build_report.json>).
Generate it with:

```bash
python -m src.build_prompts
```

The builder uses [`configs/prompts.yaml`](<./configs/prompts.yaml>) and keeps
source references in metadata where configured. Those references are useful
for later analysis, but they are not part of the model input.

## 4. Prompt filtering

Filtering reduced 4,500 raw prompts to 4,402 usable prompts and rejected 98:

| Rejection reason | Count |
|---|---:|
| Near duplicate | 42 |
| Overlap with evaluation prompt | 19 |
| Too short | 16 |
| Too long | 9 |
| Held-out HornMT evaluation pair | 5 |
| Repeated characters | 4 |
| PII or link | 2 |
| Low Ge'ez ratio | 1 |

The kept set contains 1,746 QA, 889 translation, 1,103 summarization, and
664 Wikipedia-knowledge prompts. The estimated mean prompt size is 250.6
tokens, with a maximum of 1,788 tokens.

Example kept record from
[`data/prompts/prompts_filtered.jsonl`](<./data/prompts/prompts_filtered.jsonl>):

```json
{"id":"qa-sentence_questions-00001","task":"qa",
 "prompt":"፨፨ ኮምፒዩተራችን እንዳለው እና እንደሌለው እንዴት ማወቅ እንችላለን?",
 "n_words":8,"est_tokens":28}
```

Example rejected record from
[`data/prompts/prompts_rejected.jsonl`](<./data/prompts/prompts_rejected.jsonl>):

```json
{"id":"qa-aya-00006","task":"qa","reject_reason":"too_long"}
```

The complete counts are in
[`data/prompts/filter_report.json`](<./data/prompts/filter_report.json>).
Run the filter with:

```bash
python -m src.filter_prompts
```

For exact tokenizer counts instead of the configured word-based estimate:

```bash
python -m src.filter_prompts --tokenizer-repo google/gemma-4-26B-A4B-it
```

The filter removes prompt overlap with the evaluation set and held-out HornMT
pairs, so `prompts_filtered.jsonl` is the intended input to teacher
generation.

## 5. Next stages: generate teacher data

### 5.1 Configure and smoke-test the teacher

Before a full run, verify the model repository/backend and, for GGUF, provide
a valid local GGUF path or repository/file in
[`configs/teachers/gemma4_26b_a4b.yaml`](<./configs/teachers/gemma4_26b_a4b.yaml>).
Run a small test first:

```bash
python -m src.generate \
  --teacher configs/teachers/gemma4_26b_a4b.yaml \
  --limit 5
```

The generator is resumable and writes teacher responses under
`data/teacher_outputs/`, with a summary under `results/02_generation/`.
Generate the complete pool:

```bash
python -m src.generate \
  --teacher configs/teachers/gemma4_26b_a4b.yaml
```

For a bounded run, task-specific generation, or multiple machines:

```bash
python -m src.generate --teacher configs/teachers/gemma4_26b_a4b.yaml --max-minutes 150
python -m src.generate --teacher configs/teachers/gemma4_26b_a4b.yaml --task translation
python -m src.generate --teacher configs/teachers/gemma4_26b_a4b.yaml --shard 0/2
python -m src.generate --teacher configs/teachers/gemma4_26b_a4b.yaml --shard 1/2
```

Do not use `--save-raw` unless reasoning blocks are needed for debugging.
Teacher responses should be quality-checked before using them as student
targets.

### 5.2 Score teacher outputs

```bash
python -m src.score_teacher \
  data/teacher_outputs/gemma4_26b_a4b.jsonl \
  --output results/02_generation/teacher_scores.json
```

Inspect empty, truncated, non-Ge'ez, and low-diversity responses before
training. The scoring metrics are diagnostics; they do not replace the
held-out evaluation.

### 5.3 Build the supervised fine-tuning data

Convert usable teacher prompt/response pairs into chat-format JSONL:

```bash
python -m src.make_sft \
  --input data/teacher_outputs \
  --teacher gemma4_26b_a4b \
  --output data/sft/teacher_sft.jsonl
```

The resulting records contain `messages` with a user prompt and assistant
teacher response. The held-out evaluation file must not be passed to this
command.

## 6. Student fine-tuning

Select a student model and fill its currently empty configuration file with a
real model repository and training settings. For example, the configuration
must define:

```yaml
student_repo: Qwen/Qwen2.5-0.5B-Instruct
output_dir: results/03_student/qwen_student
num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 2.0e-5
max_seq_length: 2048
```

Then run standard response-level SFT:

```bash
python -m src.train_student \
  --student configs/students/qwen3_5_0_8b.yaml \
  --data data/sft/teacher_sft.jsonl
```

This stage currently trains on teacher responses with Hugging Face
`Trainer`. The repository also contains
[`src/distill_loss.py`](<./src/distill_loss.py>) for a temperature-scaled
logit KD loss, but the current `train_student.py` entry point does not yet
wire a teacher-forward pass and logit-cache dataset into training. Logit KD
therefore requires a separate implementation stage; do not describe response
SFT as logit distillation.

## 7. Evaluation and reporting

After student training, add or use an evaluation runner that:

1. Reads only `prompt` from `data/evaluation.jsonl`.
2. Generates one response per evaluation item.
3. Scores translations and free text against `reference`.
4. Scores MCQs by exact option (`A-D` or `1-4`).
5. Treats news summarization as a weak-reference/manual-judge task.
6. Writes predictions and aggregate metrics under `results/04_evaluation/`.

The current [`src/evaluate.py`](<./src/evaluate.py>) implements the
`teacher_pilot` mode; its `student` mode is not yet implemented. The existing
pilot command is:

```bash
python -m src.evaluate \
  --mode teacher_pilot \
  --teacher configs/teachers/gemma4_26b_a4b.yaml
```

Before comparing students, keep the evaluation prompts and references fixed,
never train on them, and record the exact teacher/student model revisions,
generation settings, and dataset hashes.

## Repository outputs

- [`data/evaluation.jsonl`](<./data/evaluation.jsonl>): held-out prompts and scoring references.
- [`data/evaluation_report.json`](<./data/evaluation_report.json>): evaluation-set counts and warnings.
- [`data/prompts/prompts_raw.jsonl`](<./data/prompts/prompts_raw.jsonl>): 4,500 built prompts.
- [`data/prompts/prompts_filtered.jsonl`](<./data/prompts/prompts_filtered.jsonl>): 4,402 prompts eligible for teacher generation.
- [`data/prompts/prompts_rejected.jsonl`](<./data/prompts/prompts_rejected.jsonl>): 98 rejected prompts with reasons.
- [`data/prompts/build_report.json`](<./data/prompts/build_report.json>): prompt-build statistics.
- [`data/prompts/filter_report.json`](<./data/prompts/filter_report.json>): filtering statistics.
- [`results/01_teacher_pilot/fertility.csv`](<./results/01_teacher_pilot/fertility.csv>): candidate tokenizer comparison.
- `data/teacher_outputs/`: generated teacher responses after stage 5.
- `data/sft/`: chat-format teacher-response training data after stage 5.
- `results/03_student/`: student checkpoints after fine-tuning.
