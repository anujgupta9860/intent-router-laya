# Laya RLCD Fine-Tuning — Training Report
**Date:** 2026-10-06 | **Project:** innovation-lab-2026 | **VM:** laya-finetune-spot (deleted after run)

## What was trained
Laya's full decision heads (intent choice, needs_human noul, utterance-type choice, guardrail score) via RLCD — Track B from `train/README.md`. This trains the actual Score (guardrail) head, unlike the encoder track where the guardrail was heuristic.

## Infrastructure
- **GPU quota:** Approved 2026-10-06 ~00:28 EDT — `GPUS_ALL_REGIONS` 0 → 1 (case 0720b064-d3d2-48fc-b12c-d7abd59f3404)
- **VM:** `laya-finetune-spot`, us-central1-a, n1-standard-4 (4 vCPU, 15 GB), 1× NVIDIA T4 **Spot**, Ubuntu 22.04, 100 GB balanced disk
- **Why spot:** On-demand T4 and L4 were stocked out in 13 zones across 4 regions (us-central1 a/b/c/f, us-east1 b/c/d, us-west1 a/b, us-east4 c, europe-west4 a). Spot pulls from a separate capacity pool and succeeded on the first try in us-central1-a.
- **Cost:** Spot T4 ≈ $0.04–0.16/hr (60–91% off the $0.39/hr on-demand list price). Total run cost: well under $1.
- **VM deleted** after checkpoint verified — no ongoing charges.

## Dataset
- `train/build_dataset.py --repeat 20` → **580 records** (6 intents: account_update, billing_inquiry, fallback, order_status, sales_question, technical_support)
- `convert_dataset.py` → 580 RLCD rows in `{state, questions, gold}` format
- `preprocess.py` → **2,320 training sequences** (580 × 4 questions), 2,088 train / 232 held out for calibration
- Base model: `convaiinnovations/laya` (421M params, downloaded from HuggingFace)

## Training config (unchanged from upstream notebook)
- 4 epochs, micro-batch 8, grad-accum 4, GRPO group 4
- LR: encoder 2.5e-5 / heads 1e-4, sigma 0.4→0.1, cosine schedule
- Launcher: `torchrun --standalone --nproc_per_node=1` in tmux
- Spot safety: rolling `checkpoint_latest/` written every epoch, synced to GCS

## Results
| Metric | Value |
|---|---|
| Final loss (epoch 1, step 100) | 0.0412 (from 0.6379 at step 50) |
| Final reward | 0.652 (from 0.412) |
| Calibration temps (choice, score, noul) | [4.038, 1.0, 1.0] |
| Checkpoint | `gs://laya-checkpoints-anuj/laya-rlcd/laya_finetuned/` |

## Smoke test — the known failure case
**Input:** "delete my account"
- **Before (zero-shot base):** intent = `technical_support`, guardrail 1.83
- **After (fine-tuned):** intent = **`account_update`** (98.08% confidence), guardrail = 0.0 (safe, 100%), needs_human = 0.0 (100%)

The fine-tuning fixed the exact misclassification that motivated this run.

## Files
- Runbook + scripts: `~/workspace/laya-training/` (RUNBOOK.md, convert_dataset.py, preprocess.py, train_single_gpu.py)
- Screenshots: `~/workspace/your_files/laya-gpu-run/screenshots/` (quota request, T4 form, L4 form)
- Checkpoint: `gs://laya-checkpoints-anuj/laya-rlcd/laya_finetuned/`

## Next steps
1. Evaluate on the 400 held-out test cases (full accuracy/precision per intent)
2. Deploy to the Intent Router as `SYSTEM1_BACKEND=laya` with `LAYA_CHECKPOINT` pointed at the GCS checkpoint
3. Compare against the ModernBERT encoder baseline (val accuracy 1.000, temp 0.573)
