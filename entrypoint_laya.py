#!/usr/bin/env python3
"""Download the fine-tuned Laya checkpoint from GCS on container startup.

Skips download if the checkpoint directory already exists and is non-empty.
Then execs uvicorn.
"""
import os
import subprocess
import sys
from pathlib import Path

CHECKPOINT_GCS = os.environ.get(
    "CHECKPOINT_GCS", "gs://laya-checkpoints-anuj/laya-rlcd/laya_finetuned/"
)
CHECKPOINT_LOCAL = Path(
    os.environ.get("LAYA_CHECKPOINT", "/srv/app/checkpoints/laya_finetuned")
)


def main() -> None:
    if CHECKPOINT_LOCAL.exists() and any(CHECKPOINT_LOCAL.iterdir()):
        print(f"checkpoint already present at {CHECKPOINT_LOCAL}, skipping download",
              flush=True)
    else:
        print(f"downloading checkpoint from {CHECKPOINT_GCS} ...", flush=True)
        CHECKPOINT_LOCAL.mkdir(parents=True, exist_ok=True)
        from google.cloud import storage

        # Parse gs://bucket/prefix/
        assert CHECKPOINT_GCS.startswith("gs://"), CHECKPOINT_GCS
        parts = CHECKPOINT_GCS[5:].split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))
        if not blobs:
            print(f"ERROR: no blobs found at {CHECKPOINT_GCS}", flush=True)
            sys.exit(1)
        for blob in blobs:
            rel = blob.name[len(prefix):].lstrip("/")
            if not rel:
                continue
            dest = CHECKPOINT_LOCAL / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(dest))
            print(f"  downloaded {rel}", flush=True)
        print(f"checkpoint ready at {CHECKPOINT_LOCAL}", flush=True)

    port = os.environ.get("PORT", "8080")
    os.execvp(
        "uvicorn",
        ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", port],
    )


if __name__ == "__main__":
    main()
