# Amharic capability distillation

Pipeline: teacher pilot -> prompt building -> filtering -> teacher generation -> LoRA/QLoRA finetune -> evaluation.

## Stages
1. `src/fertility.py` + `src/evaluate.py` (teacher_pilot): compare candidate teachers
2. `src/build_prompts.py`: real open-licensed Amharic datasets -> prompts (with provenance), dedupe vs `data/evaluation.jsonl`
3. `src/generate.py --teacher configs/teachers/<name>.yaml`: resumable teacher generation
4. `src/filter.py`: quality filtering
5. `src/train_student.py`: LoRA/QLoRA finetune
6. `src/evaluate.py` (student) + `src/report.py`

## Results
(TODO)
