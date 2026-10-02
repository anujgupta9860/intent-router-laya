#!/usr/bin/env bash
# One-time GCP setup for the intent-router-laya POC:
#   - selects the GCP project
#   - enables required APIs (Vertex AI, GKE, Artifact Registry)
#   - creates an Artifact Registry Docker repo
#   - creates a GKE Autopilot cluster and fetches credentials
#
# Usage:
#   PROJECT=my-project REGION=us-central1 CLUSTER=intent-router-laya-cluster \
#     AR_REPO=intent-router-laya ./scripts/setup-gcp.sh
set -euo pipefail

PROJECT="${PROJECT:-my-gcp-project}"
REGION="${REGION:-us-central1}"
CLUSTER="${CLUSTER:-intent-router-laya-cluster}"
AR_REPO="${AR_REPO:-intent-router-laya}"

echo "==> Using project=${PROJECT} region=${REGION} cluster=${CLUSTER} repo=${AR_REPO}"

echo "==> Setting gcloud project"
gcloud config set project "$PROJECT"

echo "==> Enabling APIs (aiplatform, container, artifactregistry)"
gcloud services enable \
  aiplatform.googleapis.com \
  container.googleapis.com \
  artifactregistry.googleapis.com

echo "==> Creating Artifact Registry repo ${AR_REPO}"
if ! gcloud artifacts repositories describe "$AR_REPO" --location="$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$AR_REPO" \
    --repository-format=docker \
    --location="$REGION" \
    --description="intent-router-laya container images"
else
  echo "    (already exists, skipping)"
fi

echo "==> Creating GKE Autopilot cluster ${CLUSTER}"
if ! gcloud container clusters describe "$CLUSTER" --region "$REGION" >/dev/null 2>&1; then
  gcloud container clusters create-auto "$CLUSTER" \
    --region "$REGION" \
    --project "$PROJECT"
else
  echo "    (already exists, skipping)"
fi

echo "==> Fetching cluster credentials"
gcloud container clusters get-credentials "$CLUSTER" \
  --region "$REGION" \
  --project "$PROJECT"

echo ""
echo "GCP setup complete."
echo "Next: deploy a Gemma model from Vertex AI Model Garden and note the"
echo "endpoint id, then run ./scripts/deploy.sh (see README)."
