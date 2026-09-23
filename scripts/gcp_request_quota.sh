#!/bin/bash
# Request the GPU quota VLAProjects needs, via the Cloud Quotas API.
#
# Two quotas are required and BOTH must be granted or VM creation fails with an error
# that does not say which one is short:
#   - a regional preemptible L4 quota  (spot is a SEPARATE quota from on-demand)
#   - the global all-regions GPU quota
#
# This script cannot do the one thing that actually gates you: **upgrading the billing
# account from Free Trial to paid**. GPU quota is 0 on the free trial and increase
# requests are rejected outright. That upgrade needs payment details entered in the
# console. The script checks for it and stops with instructions rather than firing off
# a request that will be denied.
#
# Usage:
#   ./scripts/gcp_request_quota.sh <PROJECT_ID> [REGION] [--dry-run]
#
# Approval typically takes 1-2 business days, so run this before you need the GPU.

set -euo pipefail

PROJECT="${1:-}"
REGION="${2:-us-central1}"
DRY_RUN=""
for arg in "$@"; do [ "$arg" = "--dry-run" ] && DRY_RUN=1; done

if [ -z "$PROJECT" ] || [ "$PROJECT" = "--dry-run" ]; then
  echo "usage: $0 <PROJECT_ID> [REGION] [--dry-run]" >&2
  exit 2
fi

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mSTOP: %s\033[0m\n' "$*" >&2; exit 1; }

# The homebrew cask does not put gcloud on PATH; find it rather than making the caller
# fix their shell profile first.
if ! command -v gcloud >/dev/null 2>&1; then
  for d in /opt/homebrew/share/google-cloud-sdk/bin /usr/local/share/google-cloud-sdk/bin \
           "$HOME/google-cloud-sdk/bin"; do
    [ -x "$d/gcloud" ] && { export PATH="$d:$PATH"; break; }
  done
fi

# --------------------------------------------------------------- preflight

say "Preflight"
command -v gcloud >/dev/null || die "gcloud not installed. brew install --cask google-cloud-sdk"
gcloud components list --only-local-state --format='value(id)' 2>/dev/null | grep -qx beta \
  || { echo "installing beta component..."; gcloud components install beta --quiet; }

ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | head -1)
[ -n "$ACCOUNT" ] || die "not authenticated. Run: gcloud auth login"
echo "  account: $ACCOUNT"
echo "  project: $PROJECT"
echo "  region:  $REGION"

gcloud projects describe "$PROJECT" >/dev/null 2>&1 || die "cannot see project '$PROJECT'"

PROJECT_NUMBER=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
echo "  number:  $PROJECT_NUMBER"

# ------------------------------------------------------- the actual gate

say "Billing check (this is the real gate)"
BILLING=$(gcloud beta billing projects describe "$PROJECT" \
            --format='value(billingAccountName,billingEnabled)' 2>/dev/null || true)
if [ -z "$BILLING" ] || [[ "$BILLING" == *"False"* ]]; then
  die "billing is not enabled on this project.
  GPU quota is 0 on the Free Trial and increase requests are REJECTED.
  You must upgrade to a paid billing account first:
    https://console.cloud.google.com/billing  ->  'Upgrade' / 'Activate full account'
  Unused free-trial credit carries over and expires 90 days after signup."
fi
echo "  billing: $BILLING"
echo "  NOTE: gcloud cannot distinguish a Free Trial account from a paid one."
echo "        If the requests below are denied instantly, that is why."

say "Enabling compute API"
gcloud services enable compute.googleapis.com --project="$PROJECT"

# ------------------------------------------------- discover exact quota ids

say "Discovering quota IDs"
# The Cloud Quotas API uses its own IDs, which differ from the console metric names
# (console: PREEMPTIBLE_NVIDIA_L4_GPUS). Discover rather than hardcode, so this keeps
# working if Google renames them.
ALL_QUOTAS=$(gcloud beta quotas info list \
  --service=compute.googleapis.com --project="$PROJECT" \
  --format='value(quotaId)' 2>/dev/null || true)

[ -n "$ALL_QUOTAS" ] || die "could not list quotas (is the compute API enabled and billing on?)"

L4_QUOTA=$(echo "$ALL_QUOTAS" | grep -i "preemptible" | grep -i "l4" | head -1)
GLOBAL_QUOTA=$(echo "$ALL_QUOTAS" | grep -iE "gpus.*all.*region" | head -1)

if [ -z "$L4_QUOTA" ]; then
  echo "  could not auto-detect the preemptible L4 quota id. Candidates:" >&2
  echo "$ALL_QUOTAS" | grep -i l4 | sed 's/^/    /' >&2
  die "pass the right id manually (edit L4_QUOTA in this script)"
fi
[ -n "$GLOBAL_QUOTA" ] || die "could not auto-detect the all-regions GPU quota id"

echo "  regional: $L4_QUOTA  (dimensions: region=$REGION)"
echo "  global:   $GLOBAL_QUOTA"

say "Current values"
# Read the effective limits from the compute views, which are what VM creation actually
# consults. The Cloud Quotas API reports GPUS_ALL_REGIONS as "unset" rather than 0, so
# trusting it alone would hide the real blocker.
REGIONAL_LIMIT=$(gcloud compute regions describe "$REGION" --project="$PROJECT" \
  --format='value(quotas)' 2>/dev/null | tr ';' '\n' \
  | grep "PREEMPTIBLE_NVIDIA_L4_GPUS" | grep -oE "'limit': [0-9.]+" | grep -oE "[0-9.]+" | head -1)
GLOBAL_LIMIT=$(gcloud compute project-info describe --project="$PROJECT" \
  --format='value(quotas)' 2>/dev/null | tr ';' '\n' \
  | grep "GPUS_ALL_REGIONS" | grep -oE "'limit': [0-9.]+" | grep -oE "[0-9.]+" | head -1)

REGIONAL_LIMIT=${REGIONAL_LIMIT:-0}
GLOBAL_LIMIT=${GLOBAL_LIMIT:-0}
echo "    PREEMPTIBLE_NVIDIA_L4_GPUS ($REGION): $REGIONAL_LIMIT"
echo "    GPUS_ALL_REGIONS (global):            $GLOBAL_LIMIT"

# Only request what is actually short. A request for a quota already at the target is
# noise that slows down the one you need.
NEED_REGIONAL=1; NEED_GLOBAL=1
awk "BEGIN{exit !($REGIONAL_LIMIT >= 1)}" && NEED_REGIONAL=0
awk "BEGIN{exit !($GLOBAL_LIMIT >= 1)}" && NEED_GLOBAL=0
[ $NEED_REGIONAL -eq 0 ] && echo "    -> regional already sufficient, skipping"
[ $NEED_GLOBAL   -eq 0 ] && echo "    -> global already sufficient, skipping"

if [ $NEED_REGIONAL -eq 0 ] && [ $NEED_GLOBAL -eq 0 ]; then
  say "Nothing to request — both quotas are already >= 1."
  exit 0
fi

# --------------------------------------------------------------- request

JUSTIFICATION="Fine-tuning open vision-language-action and end-to-end autonomous driving \
models (NVIDIA Isaac GR00T N1.6, DiffusionDrive) for personal research. Need one L4 GPU \
on Spot VMs for intermittent training runs of a few hours each."

say "Submitting requests"
if [ -n "$DRY_RUN" ]; then
  echo "  DRY RUN — nothing submitted. Would request:"
  [ $NEED_REGIONAL -eq 1 ] && echo "    $L4_QUOTA = 1  (region=$REGION)"
  [ $NEED_GLOBAL   -eq 1 ] && echo "    $GLOBAL_QUOTA = 1  (global)"
  exit 0
fi

RC1=0; RC2=0
set +e
if [ $NEED_REGIONAL -eq 1 ]; then
  gcloud beta quotas preferences create \
    --project="$PROJECT" \
    --service=compute.googleapis.com \
    --quota-id="$L4_QUOTA" \
    --dimensions="region=$REGION" \
    --preferred-value=1 \
    --email="$ACCOUNT" \
    --justification="$JUSTIFICATION"
  RC1=$?
fi

if [ $NEED_GLOBAL -eq 1 ]; then
  gcloud beta quotas preferences create \
    --project="$PROJECT" \
    --service=compute.googleapis.com \
    --quota-id="$GLOBAL_QUOTA" \
    --preferred-value=1 \
    --email="$ACCOUNT" \
    --justification="$JUSTIFICATION"
  RC2=$?
fi
set -e

say "Result"
[ $NEED_REGIONAL -eq 1 ] && { [ $RC1 -eq 0 ] && echo "  regional L4 request submitted" || echo "  regional L4 request FAILED (rc=$RC1)"; }
[ $NEED_GLOBAL   -eq 1 ] && { [ $RC2 -eq 0 ] && echo "  global GPU request submitted"  || echo "  global GPU request FAILED (rc=$RC2)"; }

cat <<EOF

Check status:
  gcloud beta quotas preferences list --project=$PROJECT
Or in the console:
  https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT

Approval usually takes 1-2 business days. Once granted, create the VM with
scripts/ (see docs/gcp_setup.md) — note L4 requires the G2 machine family
(g2-standard-8), with no --accelerator flag.
EOF
