# Training your own System 1 model

Two tracks. Both end with a checkpoint the router serves as
`SYSTEM1_BACKEND=...` — no commercial API.

## Track A — Fine-tune an encoder on your intents (worked, in this repo)

Trains a ModernBERT classifier (same family as Laya's `english`
checkpoint) on your intent taxonomy, with temperature scaling so the
confidence numbers stay calibrated. Fully reproducible, no Laya
internals needed.

```bash
# 1. install training deps (CPU ok; GPU faster)
pip install torch transformers scikit-learn

# 2. build the labeled dataset from data/intents.yaml
python train/build_dataset.py --out train/dataset.jsonl --repeat 8

# 3. fine-tune
python train/finetune.py --data train/dataset.jsonl --out models/intent-encoder

# 4. serve it
SYSTEM1_BACKEND=encoder LAYA_CHECKPOINT=models/intent-encoder \
  uvicorn src.app:app --port 8080
```

What you get in `models/intent-encoder/`: the fine-tuned weights,
tokenizer, and `calibration.json` (fitted temperature + label map + val
accuracy). The router loads it via `src/encoder_client.py`.

**Honest scope:** the encoder learns the *intent*. `needs_human`,
`utterance_type`, and `guardrail_score` are derived heuristically in
this track (see `src/encoder_client.py`). For production traffic,
replace the template-augmented dataset with real labeled queries —
that's where most of the accuracy comes from, not the architecture.

## Track B — Fine-tune Laya itself (RLCD)

Laya's own fine-tuning flow trains the actual decision heads (choice /
noul / score) with reinforcement learning from calibrated decisions.
Upstream provides:

- a fine-tuning notebook (Kaggle 2xT4) and an Apple Silicon script in
  the [laya repo](https://github.com/NandhaKishorM/laya) — "Fine-tune
  for better accuracy" / "Fine-Tuning" sections, which run the whole
  loop: build dataset → train with RLCD → calibrate temperatures →
  evaluate → export.

Use `train/build_dataset.py` to generate the labeled decisions in JSONL,
then reshape to their expected format:

```bash
python train/build_dataset.py --out train/dataset.jsonl --repeat 20
# -> each line: {query, intent, needs_human, utterance_type, guardrail_band}
# map to Laya's (state, questions, expected answers) triples per their notebook
```

Why bother: Laya's vendor numbers put the fine-tuned
`laya-typed-decisions` checkpoint at 0.766 accuracy vs 0.362 for the base
checkpoint on their 2,000-decision benchmark — fine-tuning is where the
accuracy lives. And unlike Track A, the Score (guardrail) head is
*trained*, not heuristic.

Serve the fine-tuned checkpoint by pointing `LAYA_CHECKPOINT` at it:

```bash
SYSTEM1_BACKEND=laya LAYA_CHECKPOINT=/path/to/my-laya-finetune \
  uvicorn src.app:app --port 8080
```

## Which track?

| | Track A: encoder fine-tune | Track B: Laya RLCD fine-tune |
|---|---|---|
| What you train | intent classifier head | full decision heads (choice/noul/score) |
| Tooling | plain HF Trainer, in this repo | Laya's notebook/scripts |
| Guardrail | heuristic | learned |
| Effort | an afternoon on CPU | GPU + their RLCD loop |
| Serves as | `SYSTEM1_BACKEND=encoder` | `SYSTEM1_BACKEND=laya` |

Start with A to prove the serving path end to end; move to B when the
guardrail needs to be learned rather than heuristic.
