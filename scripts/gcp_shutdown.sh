#!/bin/bash
# GCE shutdown script — runs on spot preemption (~30s budget) and on manual stop.
#
# common/trainer.py traps SIGTERM, finishes the in-flight step, and writes an atomic
# checkpoint. All this script has to do is deliver that signal, wait for the write to
# land, and get the result off a disk that may be about to disappear.
#
# Installed via: --metadata-from-file=shutdown-script=scripts/gcp_shutdown.sh
# Logs: journalctl -u google-shutdown-scripts

set -uo pipefail

BUCKET="${VLA_BUCKET:-}"          # e.g. gs://my-bucket-vla ; set via instance metadata
CKPT_DIR="${VLA_CKPT_DIR:-/home/*/VLAProjects/ckpt}"
GRACE_SECONDS=20                  # of the ~30s GCE allows; leaves room for the sync

log() { echo "[shutdown] $*" | systemd-cat -t vla-shutdown 2>/dev/null || echo "[shutdown] $*"; }

# 1. Signal every trainer process. pkill -f matches the full command line, so this
#    catches `python train.py ...` regardless of how it was launched.
mapfile -t PIDS < <(pgrep -f "python.*train" || true)

if [ ${#PIDS[@]} -eq 0 ]; then
  log "no training process found; nothing to checkpoint"
else
  log "SIGTERM -> ${PIDS[*]}"
  kill -TERM "${PIDS[@]}" 2>/dev/null

  # 2. Wait for a clean exit, but never past the grace budget — being killed
  #    mid-sync is worse than syncing a slightly older checkpoint.
  for _ in $(seq "$GRACE_SECONDS"); do
    sleep 1
    if ! kill -0 "${PIDS[@]}" 2>/dev/null; then
      log "trainer exited cleanly"
      break
    fi
  done
fi

# 3. Push checkpoints to GCS. The boot disk survives --instance-termination-action=STOP,
#    so this is belt-and-braces; it is what saves you if the instance is ever deleted.
if [ -n "$BUCKET" ]; then
  for d in $CKPT_DIR; do
    [ -d "$d" ] || continue
    log "syncing $d -> $BUCKET/ckpt"
    timeout 8 gcloud storage rsync -r "$d" "$BUCKET/ckpt" 2>&1 | tail -3
  done
else
  log "VLA_BUCKET unset; skipping GCS sync (checkpoints remain on the boot disk)"
fi

log "done"
