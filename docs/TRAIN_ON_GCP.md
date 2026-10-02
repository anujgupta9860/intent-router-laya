# Train a Laya Model on GCP — Console Walkthrough

Step-by-step for training in Google Cloud using only the web console.
Two paths, pick one:

- **Path A — Vertex AI Workbench + Laya's RLCD notebook.** Fine-tunes the
  actual Laya decision heads (`laya-typed-decisions`). This is the track
  that learns intent *and* the guardrail score head. Best accuracy.
- **Path B — Compute Engine Deep Learning VM + our encoder track.**
  Fine-tunes a ModernBERT intent classifier (`train/finetune.py` from this
  repo). Simpler, faster, intent-only (guardrail stays heuristic).

> **Screenshots:** this guide was written without access to your GCP
> project, so every step lists exactly what to capture. Work through it
> once, save the screenshots, and they can be embedded here later —
> see the checklist at the end.

## 0. Prerequisites (do once)

1. A GCP project with **billing enabled**.
   Console: top bar project picker → *New project* (or select existing).
2. **GPU quota** — the #1 gotcha. New projects usually have **0 GPU quota**,
   so instance creation will fail until you raise it:
   - Console: *IAM & Admin → Quotas*
   - Filter: `NVIDIA T4 GPUs`, region `us-central1`
     (also check `GPUs (all regions)` if you want headroom)
   - Select the checkbox → *Edit quotas* → request e.g. **2** → submit.
   - Approval is usually minutes to a few hours. You cannot proceed
     without this.
3. A GCS bucket for the finished checkpoint (create it now or later):
   *Cloud Storage → Buckets → Create* (name like `laya-checkpoints-<you>`,
   region `us-central1`).

## Path A — Vertex AI Workbench + Laya RLCD notebook

The Laya project ships `laya_finetune_typed_decisions_2xT4_kaggle.ipynb`
(notebooks/ in github.com/NandhaKishorM/laya). It was written for Kaggle's
2×T4, but runs on a GCP Workbench instance with 1–2 T4s.

### A1. Create the Workbench instance

Console: **Vertex AI → Workbench → Instances → Create new**

| Field | Value |
|---|---|
| Name | `laya-train` |
| Region | `us-central1` |
| Zone | `us-central1-a` (T4 availability is good here) |
| Machine type | `n1-standard-4` (bump to `n1-standard-8` for 2×T4) |
| GPUs | Check *Add GPUs* → type **NVIDIA T4**, count **1** (or 2 if your quota allows — the notebook is tuned for 2) |

Leave the boot disk / environment as default, click **Create**, wait for
the green check.

📸 *Screenshot: Workbench instance list showing `laya-train` running.*

### A2. Open JupyterLab and get the code

Click **Open JupyterLab** on the instance → *File → New → Terminal*:

```bash
git clone https://github.com/NandhaKishorM/laya.git
git clone https://github.com/anujgupta9860/intent-router-laya.git
pip install -U laya "huggingface_hub[cli]"
```

In JupyterLab's file browser, open
`laya/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`.

📸 *Screenshot: JupyterLab with the notebook open.*

### A3. Prepare your training data

The notebook builds its own dataset, but to train on **your** intents,
generate the labeled decisions from this repo first (in the terminal):

```bash
cd intent-router-laya
python train/build_dataset.py --out train/dataset.jsonl --repeat 20
```

Each line is `{query, intent, needs_human, utterance_type,
guardrail_band}` — reshape these into the notebook's expected
`(state, questions, expected answers)` triples (the notebook documents
the format in its data-loading cell).

### A4. Run the notebook

Run all cells. The loop is: **build dataset → train with RLCD →
calibrate temperatures → evaluate → export**. On 1×T4, if you hit CUDA
OOM, halve the batch-size variable in the training cell and re-run.

Expected: the notebook prints accuracy before/after — upstream reports
~0.36 → ~0.77 on their 2,000-decision benchmark; your numbers will differ
on your data.

📸 *Screenshot: notebook's final evaluation cell with accuracy numbers.*

### A5. Export the checkpoint to GCS

In the terminal (notebook's export cell writes to a local dir, e.g.
`./laya-finetuned/`):

```bash
gcloud storage cp -r ./laya-finetuned gs://<your-bucket>/laya-finetuned/
```

### A6. Stop the instance

**Vertex AI → Workbench → select `laya-train` → Stop** (or Delete if
you're done — stops the GPU meter; the disk keeps a small charge until
deleted).

---

## Path B — Compute Engine VM + encoder track

Simpler path: trains our ModernBERT intent classifier with plain
HF Trainer. Good for proving the serving path end to end.

### B1. Create the GPU VM

Console: **Compute Engine → VM instances → Create instance**

| Field | Value |
|---|---|
| Name | `laya-encoder-train` |
| Region / zone | `us-central1` / `us-central1-a` |
| Machine type | `n1-standard-4` |
| GPUs | *Add GPU* → **NVIDIA T4** × 1 (needs the quota from step 0) |
| Boot disk | *Change* → search **Deep Learning** → pick the **PyTorch + CUDA**
  image (torch/CUDA preinstalled, saves ~10 min) |

Click **Create**.

📸 *Screenshot: VM instance list showing `laya-encoder-train` running.*

### B2. SSH and train

Click **SSH** on the instance row (browser SSH), then:

```bash
git clone https://github.com/anujgupta9860/intent-router-laya.git
cd intent-router-laya
pip install transformers scikit-learn accelerate pyyaml "huggingface_hub[cli]"
# torch + CUDA already on the Deep Learning image

# 1. build the labeled dataset from data/intents.yaml (no network needed)
python train/build_dataset.py --out train/dataset.jsonl --repeat 8

# 2. download the base model first (cleaner than in-script download)
hf download answerdotai/ModernBERT-base

# 3. train offline — 5 epochs, a few minutes on a T4
HF_HUB_OFFLINE=1 python train/finetune.py \
  --data train/dataset.jsonl --out models/intent-encoder
```

You should see `val accuracy` and `fitted temperature` printed, and
`models/intent-encoder/` containing weights, tokenizer, and
`calibration.json`.

📸 *Screenshot: terminal showing `val accuracy: …` and the export line.*

### B3. Export to GCS and stop the VM

```bash
gcloud storage cp -r models/intent-encoder gs://<your-bucket>/intent-encoder/
```

Then **Compute Engine → VM instances → select → Stop** (or Delete).

---

## Serve the trained checkpoint

Download the checkpoint anywhere and point the router at it:

```bash
# Path A (fine-tuned Laya):
gcloud storage cp -r gs://<your-bucket>/laya-finetuned ./laya-finetuned
SYSTEM1_BACKEND=laya LAYA_CHECKPOINT=./laya-finetuned \
  uvicorn src.app:app --port 8080

# Path B (encoder):
gcloud storage cp -r gs://<your-bucket>/intent-encoder ./models/intent-encoder
SYSTEM1_BACKEND=encoder LAYA_CHECKPOINT=./models/intent-encoder \
  uvicorn src.app:app --port 8080
```

## Cost notes (approximate — check the pricing calculator)

| Resource | Approx. on-demand |
|---|---|
| 1× NVIDIA T4 | ~$0.35–0.65/hr depending on region |
| n1-standard-4 (4 vCPU) | ~$0.19/hr |
| Persistent disk | ~$0.04/GB/month |

A full fine-tune run is typically a few GPU-hours. **Stop/Delete the
instance/VM when done** — a running GPU idles at full price.

## Troubleshooting

- **`Quota exceeded` on create** — you skipped step 0. Request the quota
  and wait for approval.
- **CUDA OOM (Path A)** — halve the batch size in the notebook's training
  cell; the notebook targets 2×T4.
- **`hf: command not found`** — `pip install -U "huggingface_hub[cli]"`,
  then re-open the terminal.
- **Slow first run** — checkpoint download + cold model load; subsequent
  runs use the HF cache.
- **No proxy/CA issues here** — unlike a corporate laptop, GCP VMs reach
  Hugging Face directly, so the certificate workarounds in
  `train/README.md` are not needed.

## Screenshot checklist

Capture these while walking through, then they can be embedded above:

- [ ] Quotas page showing approved NVIDIA T4 quota in us-central1
- [ ] Workbench *Create instance* form (Path A) with GPU section filled
- [ ] Workbench instance list: `laya-train` running
- [ ] JupyterLab with the fine-tuning notebook open
- [ ] Notebook evaluation cell with before/after accuracy
- [ ] Compute Engine *Create instance* form (Path B) with GPU + DL image
- [ ] VM list: `laya-encoder-train` running
- [ ] Terminal: `val accuracy` / `fitted temperature` / export line
- [ ] GCS bucket showing the uploaded checkpoint
- [ ] Router `/route` response using the trained checkpoint

---
*See also: `train/README.md` (both training tracks), `docs/RUN_LOCALLY.md`
(local run), `docs/LAYA_ARCHITECTURE.md` (how Laya works).*
