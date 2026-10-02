"""Fine-tune your own intent classifier and export it as a service-ready checkpoint.

What this trains: a ModernBERT encoder (same family as Laya's `english`
checkpoint) with a classification head over your intent taxonomy, plus
temperature scaling on a held-out split so the confidence numbers are
calibrated (the property that makes the router's gates trustworthy).

This is the "train existing model" track that needs no Laya internals:
standard Hugging Face Trainer, reproducible end to end. The exported
checkpoint plugs into the router as SYSTEM1_BACKEND=encoder - the same
SystemOneDecision shape, the same hybrid policy.

For Laya's own RLCD fine-tuning flow (their notebook/scripts), use
train/build_dataset.py to generate the labeled decisions and follow
train/README.md.

Usage:
    python train/build_dataset.py --out train/dataset.jsonl
    python train/finetune.py --data train/dataset.jsonl --out models/intent-encoder
    SYSTEM1_BACKEND=encoder LAYA_CHECKPOINT=models/intent-encoder uvicorn src.app:app

Requirements: torch, transformers, scikit-learn, accelerate, pyyaml
    pip install torch transformers scikit-learn accelerate pyyaml
A CPU can train this (small data, small model); a GPU is faster.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_ID = "answerdotai/ModernBERT-base"


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="train/dataset.jsonl")
    parser.add_argument("--out", default="models/intent-encoder")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import numpy as np
    import torch
    from sklearn.model_selection import train_test_split
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    records = load_records(Path(args.data))
    labels = sorted({r["intent"] for r in records})
    label2id = {l: i for i, l in enumerate(labels)}
    print(f"{len(records)} records, {len(labels)} labels: {labels}")

    train_recs, val_recs = train_test_split(
        records, test_size=0.2, random_state=args.seed, stratify=[r["intent"] for r in records]
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def encode(recs):
        enc = tokenizer([r["query"] for r in recs], truncation=True,
                        padding=True, max_length=128)
        enc["labels"] = [label2id[r["intent"]] for r in recs]
        return enc

    train_enc, val_enc = encode(train_recs), encode(val_recs)

    class DictDataset(torch.utils.data.Dataset):
        def __init__(self, enc):
            self.enc = enc

        def __len__(self):
            return len(self.enc["input_ids"])

        def __getitem__(self, i):
            return {k: torch.tensor(v[i]) for k, v in self.enc.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels),
        id2label={i: l for l, i in label2id.items()},
        label2id=label2id,
    )

    out = Path(args.out)
    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=args.seed,
        report_to="none",
    )
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=DictDataset(train_enc),
                      eval_dataset=DictDataset(val_enc))
    trainer.train()

    # --- temperature scaling on the validation split (calibration) ---
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for i in range(0, len(val_enc["input_ids"]), 32):
            batch = {k: torch.tensor(v[i:i + 32]) for k, v in val_enc.items()
                     if k != "labels"}
            logits_list.append(model(**batch).logits)
            labels_list.append(torch.tensor(val_enc["labels"][i:i + 32]))
    logits = torch.cat(logits_list)
    targets = torch.cat(labels_list)

    temperature = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    temp = float(temperature.detach())
    print(f"fitted temperature: {temp:.3f}")

    # --- eval report ---
    with torch.no_grad():
        scaled = logits / temp
        preds = scaled.argmax(dim=-1)
        acc = (preds == targets).float().mean().item()
        conf = torch.softmax(scaled, dim=-1).max(dim=-1).values.mean().item()
    print(f"val accuracy: {acc:.3f} | mean confidence: {conf:.3f}")

    # --- export ---
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "calibration.json").write_text(json.dumps({
        "temperature": temp,
        "labels": labels,
        "val_accuracy": acc,
        "base_model": args.model,
    }, indent=2))
    print(f"exported -> {out} (SYSTEM1_BACKEND=encoder LAYA_CHECKPOINT={out})")


if __name__ == "__main__":
    main()
