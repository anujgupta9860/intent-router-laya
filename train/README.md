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
#    accelerate is required by HF Trainer; pyyaml by build_dataset.py
pip install torch transformers scikit-learn accelerate pyyaml

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

### Downloading the base model separately (corporate networks)

`finetune.py` downloads `answerdotai/ModernBERT-base` (~600MB) on first
run via `from_pretrained`. If that download fails — e.g. SSL
certificate errors behind a corporate TLS-inspecting proxy — fetch it
with the Hugging Face CLI and train offline:

```bash
pip install -U huggingface_hub   # provides the `hf` CLI

# if the download fails with certificate errors, point Python at your
# company's root CA first (get the .pem from IT or export it from
# Keychain Access on macOS):
# export SSL_CERT_FILE=/path/to/company-root-ca.pem
# export REQUESTS_CA_BUNDLE=/path/to/company-root-ca.pem

hf download answerdotai/ModernBERT-base

HF_HUB_OFFLINE=1 python train/finetune.py \
  --data train/dataset.jsonl --out models/intent-encoder
```

`HF_HUB_OFFLINE=1` forces `from_pretrained` to resolve from the local
cache (`~/.cache/huggingface`, override with `HF_HOME`) and never touch
the network. The dataset build step (`build_dataset.py`) needs no
network at all.

## Track B — Fine-tune Laya itself (RLCD)

Laya's own fine-tuning flow trains the actual decision heads (choice /
noul / score) with reinforcement learning from calibrated decisions.

**On Apple Silicon (local, MPS)** — one command does the whole loop
(dataset → RLCD → preprocess → 4-epoch RLCD train → calibrate → export):

```bash
bash scripts/laya-stack.sh train laya
# output: checkpoints/laya_finetuned/ (exactly where `run` looks for it)
SYSTEM1_BACKEND=laya bash scripts/laya-stack.sh run
```

Takes ~1–2 hours. A rolling checkpoint is written after every epoch, so
an interruption loses at most one epoch. If you run out of memory, lower
the micro-batch: `MICRO_BATCH=4 bash scripts/laya-stack.sh train laya`.

**On CUDA** — the same loop via the DDP script (from the repo root):

```bash
python train/build_dataset.py --out train/dataset.jsonl --repeat 14
python train/convert_dataset.py --in train/dataset.jsonl --out train/laya_rlcd.jsonl
python train/preprocess.py --data train/laya_rlcd.jsonl --out train_items.pt
TRAIN_ITEMS=train_items.pt torchrun --standalone --nproc_per_node=1 \
  train/train_single_gpu.py "$(cat model_dir.txt)" ./laya_finetuned
```

Upstream also provides a fine-tuning notebook (Kaggle 2xT4) and an Apple
Silicon script in the [laya repo](https://github.com/NandhaKishorM/laya)
— "Fine-tune for better accuracy" / "Fine-Tuning" sections.

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
