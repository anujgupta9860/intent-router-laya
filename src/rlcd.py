"""On-demand RLCD retraining: export approved feedback, launch training.

The serving container has no GPU, so ``POST /rlcd/train`` works in two
modes:

- ``mode=local``: spawns a background thread running ``train/finetune.py``
  on the exported dataset. Fine for small-data CPU demos; slow for real
  checkpoints.
- ``mode=spot-vm`` (default): exports the dataset and returns the exact
  ``gcloud`` commands to fine-tune on a spot T4 VM, following
  ``docs/TRAIN_ON_GCP.md``. The VM does the heavy lifting; the service
  just prepares the data.

Job state is kept in memory (single-process). For production, back this
with a proper queue (Cloud Tasks / Pub/Sub) and Vertex AI Custom Jobs.
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time
import uuid
from pathlib import Path

log = logging.getLogger(__name__)

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _set(job_id: str, **fields) -> None:
    with _jobs_lock:
        _jobs.setdefault(job_id, {}).update(fields)


def get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        return dict(_jobs.get(job_id, {}))


def list_jobs() -> list[dict]:
    with _jobs_lock:
        return [dict(j) for j in _jobs.values()]


def _run_local_training(job_id: str, dataset: str, out_dir: str,
                        epochs: float) -> None:
    """Background thread: run train/finetune.py, capture the outcome."""
    _set(job_id, status="running", started_ts=time.time())
    try:
        proc = subprocess.run(
            ["python", "train/finetune.py", "--data", dataset,
             "--out", out_dir, "--epochs", str(epochs)],
            capture_output=True, text=True, timeout=6 * 3600,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        _set(job_id, status="done" if proc.returncode == 0 else "failed",
             returncode=proc.returncode,
             stdout_tail=proc.stdout[-4000:],
             stderr_tail=proc.stderr[-4000:],
             finished_ts=time.time())
    except Exception as exc:  # noqa: BLE001
        log.exception("local training job %s failed", job_id)
        _set(job_id, status="failed", error=str(exc),
             finished_ts=time.time())


def start_training_job(dataset: str, out_dir: str, mode: str = "spot-vm",
                       epochs: float = 4.0) -> dict:
    """Export is done by the caller; this launches (or describes) training."""
    job_id = uuid.uuid4().hex[:12]
    job = {"job_id": job_id, "mode": mode, "dataset": dataset,
           "out_dir": out_dir, "epochs": epochs,
           "status": "queued", "created_ts": time.time()}
    with _jobs_lock:
        _jobs[job_id] = job

    if mode == "local":
        t = threading.Thread(target=_run_local_training,
                             args=(job_id, dataset, out_dir, epochs),
                             daemon=True)
        t.start()
        _set(job_id, status="started")
        return {"job_id": job_id, "mode": "local", "status": "started",
                "detail": "training in background thread; poll GET /rlcd/train/{job_id}"}

    # spot-vm: hand back the exact commands (docs/TRAIN_ON_GCP.md flow).
    _set(job_id, status="awaiting_vm")
    return {
        "job_id": job_id, "mode": "spot-vm", "status": "awaiting_vm",
        "detail": (
            "Dataset exported. Run these on a spot T4 VM "
            "(see docs/TRAIN_ON_GCP.md):"
        ),
        "commands": [
            f"gcloud compute instances create laya-rlcd-train --project=innovation-lab-2026 "
            f"--zone=us-central1-a --machine-type=n1-standard-4 "
            f"--accelerator=type=nvidia-tesla-t4,count=1 --preemptible "
            f"--image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud "
            f"--boot-disk-size=100GB",
            f"gcloud compute scp {dataset} laya-rlcd-train:~/train/ --zone=us-central1-a",
            "gcloud compute ssh laya-rlcd-train --zone=us-central1-a -- "
            "'pip install torch transformers scikit-learn accelerate pyyaml && "
            f"python train/finetune.py --data ~/train/{Path(dataset).name} "
            f"--out ~/laya_finetuned_v2 --epochs {epochs}'",
            "gcloud compute scp --recurse laya-rlcd-train:~/laya_finetuned_v2 ./",
            "gsutil -m cp -r ./laya_finetuned_v2 "
            "gs://laya-checkpoints-anuj/laya-rlcd-v2/",
            "gcloud compute instances delete laya-rlcd-train --zone=us-central1-a --quiet",
        ],
    }
