#!/bin/bash
# GCE startup script — runs on every boot, including resume after preemption.
#
# Its main job is the idle-shutdown watchdog. A budget alert only emails you; this is
# what actually stops a forgotten VM from eating the $300.
#
# Installed via: --metadata-from-file=startup-script=scripts/gcp_startup.sh
# Logs: journalctl -u google-startup-scripts

set -uo pipefail

IDLE_MINUTES="${VLA_IDLE_MINUTES:-30}"
UTIL_THRESHOLD=5        # percent GPU utilisation counted as "working"
POLL_SECONDS=60

log() { echo "[startup] $*" | systemd-cat -t vla-startup 2>/dev/null || echo "[startup] $*"; }

log "boot: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no GPU visible')"

# Idle watchdog: power off after IDLE_MINUTES of GPU utilisation below threshold.
# Runs detached so the startup script itself can return.
cat >/usr/local/bin/vla-idle-watchdog <<WATCHDOG
#!/bin/bash
idle=0
need=\$(( $IDLE_MINUTES * 60 / $POLL_SECONDS ))
while true; do
  sleep $POLL_SECONDS
  util=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ -z "\$util" ] && continue
  if [ "\$util" -lt $UTIL_THRESHOLD ]; then
    idle=\$(( idle + 1 ))
  else
    idle=0
  fi
  if [ "\$idle" -ge "\$need" ]; then
    logger -t vla-startup "idle ${IDLE_MINUTES}m at <${UTIL_THRESHOLD}% GPU; powering off"
    # Triggers the shutdown script, so any running trainer still checkpoints first.
    shutdown -h now
  fi
done
WATCHDOG
chmod +x /usr/local/bin/vla-idle-watchdog

# Restart-on-failure so the watchdog survives its own crashes; a dead watchdog is a
# silently unprotected instance.
cat >/etc/systemd/system/vla-idle-watchdog.service <<UNIT
[Unit]
Description=VLA idle shutdown watchdog
After=network.target

[Service]
ExecStart=/usr/local/bin/vla-idle-watchdog
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now vla-idle-watchdog
log "idle watchdog armed: power off after ${IDLE_MINUTES}m below ${UTIL_THRESHOLD}% GPU"
