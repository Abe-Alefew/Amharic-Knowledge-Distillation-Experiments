"""Train a student model on the prepared teacher-response SFT dataset."""

import argparse
from pathlib import Path

from .utils import ROOT, load_yaml, read_jsonl


DEFAULTS = {
    "student_repo": "",
    "output_dir": "results/03_student/model",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-5,
    "max_seq_length": 2048,
    "logging_steps": 10,
    "save_steps": 500,
}


def load_student_config(path):
    config = dict(DEFAULTS)
    if path:
        path = Path(path)
        loaded = load_yaml(ROOT / path if not path.exists() else path) or {}
        config.update(loaded)
    return config


def _text_from_record(tokenizer, record):
    messages = record.get("messages")
    if messages and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return f"User: {record.get('prompt', '')}\nAssistant: {record.get('response', '')}"


def train(config, dataset_path):
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    repo = config.get("student_repo") or config.get("hf_repo")
    if not repo:
        raise ValueError("student config must define student_repo or hf_repo")
    rows = read_jsonl(dataset_path)
    if not rows:
        raise ValueError(f"training dataset is empty: {dataset_path}")

    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype="auto")
    raw = Dataset.from_list([{"text": _text_from_record(tokenizer, row)} for row in rows])
    tokenized = raw.map(
        lambda batch: tokenizer(
            batch["text"], truncation=True, max_length=int(config["max_seq_length"])
        ),
        batched=True,
        remove_columns=["text"],
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=str(ROOT / config["output_dir"]),
        num_train_epochs=float(config["num_train_epochs"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        logging_steps=int(config["logging_steps"]),
        save_steps=int(config["save_steps"]),
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    return training_args.output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True, help="student YAML")
    parser.add_argument("--data", default="data/sft/teacher_sft.jsonl")
    args = parser.parse_args()
    output = train(load_student_config(args.student), ROOT / args.data)
    print(f"saved student -> {output}")


if __name__ == "__main__":
    main()