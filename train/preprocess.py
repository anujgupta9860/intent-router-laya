"""Preprocess our RLCD JSONL into tokenized training items.

Adapted from cell 3 of laya's fine-tuning notebook
(notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb), with two changes:
  1. reads our local train/laya_rlcd.jsonl instead of the HF
     LocalLLaMA/typed-decisions dataset
  2. saves to ./train_items.pt instead of /kaggle/working/

Our gold rows use the same {probabilities, label} shape the notebook's
build_training_item() expects, so the target-building logic is unchanged.

Usage:
    python preprocess.py --data train/laya_rlcd.jsonl --out train_items.pt
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from laya.agent import _fix_tokenizer_config
from laya.common import build_sequence, render_options, QTYPES

MODEL_ID = "convaiinnovations/laya"


def build_training_item(tok, cfg, state, q, gold_q):
    t = q["type"]
    crit = q.get("criteria", {})
    if t == "choice":
        keys = list(crit.keys())
        target = [gold_q["probabilities"].get(k, 0.0) for k in keys]
    elif t == "noul":
        target = [gold_q["probabilities"].get("false", 0.5),
                  gold_q["probabilities"].get("true", 0.5)]
    elif t == "score":
        n_levels = len(crit) if isinstance(crit, list) else 4
        target = [gold_q["probabilities"].get(str(i), 0.0)
                  for i in range(n_levels)]
    else:
        return None

    s = sum(target)
    target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
    label = target.index(max(target))
    k = len(render_options({"t": t, "crit": crit}))

    seq, markers = build_sequence(
        tok, state,
        {"t": t, "ins": q["instructions"], "crit": crit},
        cfg["max_len"], cfg["head_max_len"],
    )
    if len(markers) != k:
        return None
    return {"ids": seq, "markers": markers, "qtype": QTYPES[t],
            "target": target, "label": label}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train/laya_rlcd.jsonl")
    ap.add_argument("--out", default="train_items.pt")
    ap.add_argument("--model", default=MODEL_ID)
    args = ap.parse_args()

    print(f"Fetching tokenizer and config from {args.model}...")
    model_dir = snapshot_download(args.model)
    _fix_tokenizer_config(model_dir)

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)

    print(f"Loading {args.data}...")
    items = []
    skipped = 0
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            state, questions, gold = row["state"], row["questions"], row["gold"]
            for qid, q in questions.items():
                if qid in gold:
                    it = build_training_item(tok, cfg, state, q, gold[qid])
                    if it:
                        items.append(it)
                    else:
                        skipped += 1

    print(f"Preprocessed {len(items)} training sequences ({skipped} skipped).")
    torch.save(items, args.out)
    print(f"Saved to {args.out}")
    # also stash the model dir path for the training script
    Path("model_dir.txt").write_text(model_dir)
    print(f"model_dir={model_dir}")


if __name__ == "__main__":
    main()
