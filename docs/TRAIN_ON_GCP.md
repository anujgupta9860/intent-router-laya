# Train a Laya Model on GCP — Console Walkthrough

Step-by-step for training in Google Cloud using only the web console.
Two paths, pick one:

- **Path A — Vertex AI Workbench + Laya's RLCD notebook.** Fine-tunes the
  actual Laya decision heads (`laya-typed-decisions`). This is the track
  that learns intent *and* the guardrail score head. Best accuracy.
- **Path B — Compute Engine VM + our encoder track.**
  Fine-tunes a ModernBERT intent classifier (`train/finetune.py` from this
  repo). Simpler, intent-only (guardrail stays heuristic). Runs on a plain
  CPU VM — no GPU quota needed (verified 2026-10-03); a Deep Learning GPU
  VM is faster if you have the quota.

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
HF Trainer. Good for proving the serving path end to end. The CPU route
below was verified end to end on 2026-10-03 (no GPU quota needed);
the GPU column shows the faster alternative if your quota allows.

### B1. Create the VM

Console: **Compute Engine → VM instances → Create instance**

| Field | CPU path (verified) | GPU alternative |
|---|---|---|
| Name | `laya-encoder-train` | same |
| Region / zone | `us-central1` / `us-central1-a` | same |
| Machine type | `e2-standard-4` (4 vCPU, 16 GB) | `n1-standard-4` |
| GPUs | none | *Add GPU* → **NVIDIA T4** × 1 (needs the quota from step 0) |
| Boot disk | Ubuntu 22.04 LTS, 50 GB balanced persistent disk | *Change* → search **Deep Learning** → **PyTorch + CUDA** image (torch/CUDA preinstalled, saves ~10 min) |

If the VM will upload the finished model to Cloud Storage, set
**Access scopes → Allow full access to all Cloud APIs** on the create
form (or later via Stop → Edit → Start) — the default scopes are
Storage read-only and the upload will fail without this.

Click **Create**.

📸 *Screenshot: VM instance list showing `laya-encoder-train` running.*

### B2. SSH and train

Click **SSH** on the instance row (browser SSH), then:

```bash
sudo apt-get update && sudo apt-get install -y python3-pip git
git clone https://github.com/anujgupta9860/intent-router-laya.git
cd intent-router-laya

# CPU-only torch (far smaller download than the CUDA wheel)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers scikit-learn accelerate pyyaml "huggingface_hub[cli]"
# Note: the stock Ubuntu 22.04 image ships pip 22.0.2, which does not know
# --break-system-packages — plain `pip install` just works.

# 1. build the labeled dataset from data/intents.yaml (no network needed)
python3 train/build_dataset.py --out train/dataset.jsonl --repeat 8
# -> wrote 232 records -> train/dataset.jsonl (6 intents)

# 2. download the base model first (~3 GB; cleaner than in-script download)
hf download answerdotai/ModernBERT-base

# 3. train in the background — 5 epochs, minutes on CPU
HF_HUB_OFFLINE=1 nohup python3 train/finetune.py \
  --data train/dataset.jsonl --out models/intent-encoder > train.log 2>&1 &
tail -f train.log
```

You should see `val accuracy: 1.000 | mean confidence: 1.000`,
`fitted temperature: 0.573`, and `exported -> models/intent-encoder`
— `models/intent-encoder/` then holds the weights, tokenizer, and
`calibration.json`. (Your numbers will differ on your data.)

📸 *Screenshot: terminal showing `val accuracy` / `fitted temperature`.*

Gotchas found the hard way:
- `models/intent-encoder/checkpoints/` keeps every epoch's artifacts
  (~8 GB) — it is not needed for inference. Delete it before a Docker
  build or if the disk fills up.
- If SSH-in-browser drops mid-session, the `nohup` training keeps
  running — just reconnect and `tail train.log`.

### B3. Export to GCS and stop the VM

From the same SSH session (the VM's service account authenticates;
this is why step B1 widened the access scopes):

```bash
pip install google-cloud-storage
python3 - <<'EOF'
from google.cloud import storage
import os
client = storage.Client(project="<your-project-id>")
bucket = client.create_bucket("laya-checkpoints-<you>", location="us-central1")
src = "models/intent-encoder"
n = 0
for root, _, files in os.walk(src):
    for f in files:
        p = os.path.join(root, f)
        bucket.blob("intent-encoder/" + os.path.relpath(p, src)).upload_from_filename(p)
        n += 1
print(f"uploaded {n} files")
EOF
```

(If the bucket name is taken, pick another suffix and retry.)

Then **Compute Engine → VM instances → select → Stop** (or **Delete**
— ends the VM and disk charges entirely).

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
