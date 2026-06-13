#!/usr/bin/env bash
# Download files from a RunPod network volume via S3 without aws s3 sync.
#
# RunPod's S3 API breaks on recursive ListObjects (duplicate ContinuationToken)
# and often returns intermittent 403/429 on individual GetObject calls.
# This script lists one directory level at a time, retries failed downloads,
# and skips files already on disk.
#
# Usage:
#   export RUNPOD_S3_ENDPOINT=https://s3api-eu-ro-1.runpod.io
#   export RUNPOD_S3_REGION=eu-ro-1
#   export RUNPOD_S3_BUCKET=gxksig1cul
#   bash scripts/runpod_s3_download.sh tile-upscaler/out/sheets ./out-from-runpod/sheets
#
# Prefix is relative to the bucket root (no s3://, no leading slash).
set -uo pipefail

ENDPOINT="${RUNPOD_S3_ENDPOINT:?Set RUNPOD_S3_ENDPOINT, e.g. https://s3api-eu-ro-1.runpod.io}"
REGION="${RUNPOD_S3_REGION:?Set RUNPOD_S3_REGION, e.g. eu-ro-1}"
BUCKET="${RUNPOD_S3_BUCKET:?Set RUNPOD_S3_BUCKET, e.g. gxksig1cul}"
MAX_RETRIES="${RUNPOD_S3_RETRIES:-5}"
RETRY_PAUSE="${RUNPOD_S3_PAUSE:-3}"   # seconds between retries / downloads

REMOTE_PREFIX="${1:?Usage: $0 <remote-prefix> <local-dir>}"
LOCAL_DIR="${2:?Usage: $0 <remote-prefix> <local-dir>}"
REMOTE_PREFIX="${REMOTE_PREFIX#/}"
REMOTE_PREFIX="${REMOTE_PREFIX%/}"

AWS=(aws s3 --region "$REGION" --endpoint-url "$ENDPOINT")
if [[ -n "${AWS_PROFILE:-}" ]]; then
  AWS+=(--profile "$AWS_PROFILE")
fi

FAILED=()

cp_one() {
  local key="$1"
  local dest="$2"
  if [[ -f "$dest" && -s "$dest" ]]; then
    echo "  skip (exists) s3://${BUCKET}/${key}"
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  local attempt=1
  local backoff="$RETRY_PAUSE"
  while (( attempt <= MAX_RETRIES )); do
    echo "  cp s3://${BUCKET}/${key} (attempt ${attempt}/${MAX_RETRIES})"
    if "${AWS[@]}" cp "s3://${BUCKET}/${key}" "$dest"; then
      sleep "$RETRY_PAUSE"
      return 0
    fi
    echo "  [warn] failed, retrying in ${backoff}s..." >&2
    rm -f "$dest"
    sleep "$backoff"
    backoff=$((backoff * 2))
    (( attempt++ )) || true
  done
  FAILED+=("s3://${BUCKET}/${key}")
  return 1
}

download_tree() {
  local prefix="$1"
  local local_path="$2"
  mkdir -p "$local_path"

  local listing
  if ! listing=$("${AWS[@]}" ls "s3://${BUCKET}/${prefix}/"); then
    echo "  [warn] cannot list s3://${BUCKET}/${prefix}/" >&2
    return 0
  fi

  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    if [[ "$line" =~ PRE[[:space:]]+(.+)/$ ]]; then
      download_tree "${prefix}/${BASH_REMATCH[1]}" "${local_path}/${BASH_REMATCH[1]}"
    elif [[ "$line" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}[[:space:]]+[0-9]{2}:[0-9]{2}:[0-9]{2}[[:space:]]+[0-9]+[[:space:]]+(.+)$ ]]; then
      cp_one "${prefix}/${BASH_REMATCH[1]}" "${local_path}/${BASH_REMATCH[1]}" || true
    fi
  done <<< "$listing"
}

echo "Downloading s3://${BUCKET}/${REMOTE_PREFIX}/ -> ${LOCAL_DIR}/"
download_tree "$REMOTE_PREFIX" "$LOCAL_DIR"

if ((${#FAILED[@]} > 0)); then
  echo ""
  echo "${#FAILED[@]} file(s) still failed after ${MAX_RETRIES} attempts each:"
  printf '  %s\n' "${FAILED[@]}"
  echo "Re-run the same command — already-downloaded files are skipped."
  exit 1
fi
echo "Done."
