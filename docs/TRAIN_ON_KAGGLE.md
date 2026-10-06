# Train Laya on Kaggle (Free GPUs)

Fine-tune Laya's typed-decision model on **free** Kaggle GPUs — no GCP quota,
no billing, no waiting on Google approvals.

## Why Kaggle

| | Kaggle | GCP (quota path) |
|---|---|---|
| GPUs | 2× NVIDIA T4, free | 1× T4, ~$0.35–0.65/hr |
| Weekly limit | ~30 GPU-hours free | none (pay as you go) |
| Quota needed | No | Yes — `GPUs (all regions)` (currently 0) |
| Setup | Notebook in browser | VM + drivers + env |

The Laya project ships its fine-tuning notebook **written for Kaggle**:
`laya_finetune_typed_decisions_2xT4_kaggle.ipynb`
(`notebooks/` in `github.com/NandhaKishorM/laya`). It uses DDP
(distributed data parallel) across both T4s via `torchrun`.

## Prerequisites

1. A Kaggle account: <https://www.kaggle.com> → Sign up.
2. **Phone verification** — Kaggle requires a verified phone number before it
   unlocks GPU accelerators. Profile → Settings → verify. This is the one
   step people miss.

## Steps

### 1. Create a GPU notebook

- Kaggle → **Create** → **New Notebook**.
- Right panel → **Accelerator** → select **GPU T4 x2**.
- Confirm `!nvidia-smi` shows 2× Tesla T4 (the notebook's first cell does this).

### 2. Get the training notebook

Option A — upload the Laya notebook:
```bash
# on your machine
curl -sL -o laya_finetune_kaggle.ipynb \
  https://raw.githubusercontent.com/NandhaKishorM/laya/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb
```
Then in Kaggle: **File** → **Upload Notebook** → select the `.ipynb`.

Option B — paste cell by cell into a fresh Kaggle notebook (19 cells).

### 3. Run the cells top to bottom

What each stage does:

| Cell(s) | What happens |
|---|---|
| 1 | `!nvidia-smi` — confirms 2× T4 visible |
| 2 | `pip install laya transformers datasets safetensors huggingface_hub` |
| 3 | Downloads & preprocesses **1,200 training cases** into tokenized items on disk (both DDP ranks read from here) |
| 4 | Writes `train_ddp.py` — the DDP training loop (reward-weighted typed-decision heads) |
| 5 | `torchrun --standalone --nproc_per_node=2 train_ddp.py …` — the actual fine-tune across both GPUs |
| 6–7 | Evaluates on **400 held-out cases / 2,000 decisions**, prints metrics + comparison table vs TypeSafe Jev |

Expect the `torchrun` cell to take on the order of **tens of minutes to a few
hours** depending on epochs. Keep the browser tab open — Kaggle may idle-kill
long-disconnected sessions.

### 4. Download the checkpoint

Training writes to `/kaggle/working/laya_finetuned_typed_decisions`.
When done: right panel → **Output** → download the folder (or
`!tar -czf` it first if it's large, then download the tarball).

### 5. Serve it

Upload the checkpoint to GCS and point the router at it:

```bash
export PATH=~/google-cloud-sdk/bin:$PATH
gsutil -m cp -r laya_finetuned_typed_decisions \
  gs://laya-checkpoints-anuj/laya-finetuned/

# serve (same shape the router already expects)
SYSTEM1_BACKEND=laya LAYA_CHECKPOINT=/path/to/laya_finetuned_typed_decisions \
  uvicorn src.app:app --port 8080
```

Then re-run the eval from `docs/TESTING_LIVE_SERVICE.md` — the zero-shot
failure case (`"delete my account"` → `technical_support`, guardrail 1.83)
is your regression test. A fine-tuned model should classify it correctly
with a calibrated guardrail.

## Gotchas

- **30 hrs/week GPU budget.** The fine-tune fits comfortably, but don't leave
  idle notebooks burning it. **Shut the session down** when done.
- **Session timeouts.** Kaggle kills notebooks after ~9h wall time and on long
  disconnects. Save checkpoints to `/kaggle/working` as you go (it persists
  per session) and download promptly.
- **Internet access** must be ON in the notebook settings (needed for the
  pip installs and dataset/model downloads).
- **Phone verification** (prereqs) is the #1 reason "GPU" is greyed out.
- Keep the **comparison table vs Jev** from cell 7 — it's your evidence for
  whether the fine-tuned model is production-ready.

## Video walkthrough

`laya-finetune-kaggle.mp4` (in this folder) narrates the whole flow:
account → GPU enable → notebook → training → download → serve.
