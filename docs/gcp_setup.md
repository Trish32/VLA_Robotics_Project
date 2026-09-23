# GCP setup — L4 spot for VLAProjects training

Do these in order. Step 1 has a 1–2 business-day lead time, so start it before week 2.

---

## 1. The quota gate (start this first)

**GPU quota is zero on the free trial and cannot be raised while you are on it.** You must
upgrade to a paid billing account before any GPU will attach. Upgrading keeps your unused
$300 (it expires 90 days from signup), but from that moment overage bills to your card.

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud services enable compute.googleapis.com
```

Then upgrade to paid billing in the console (Billing → "Upgrade" / "Activate full account").

Now request **two** quotas — a regional one and a global one. Missing either fails VM
creation with a quota error that does not say which one is short:

| Quota (console name) | Scope | Ask for |
|---|---|---|
| `PREEMPTIBLE_NVIDIA_L4_GPUS` | your region, e.g. `us-central1` | 1 |
| `GPUS_ALL_REGIONS` | global | 1 |

Spot quota is **separate from on-demand**; requesting `NVIDIA_L4_GPUS` does not grant spot.

Either use the console (IAM & Admin → Quotas → filter → Edit quota → Submit request), or
run the script, which does both requests and discovers the API's quota IDs rather than
relying on the console names:

```bash
./scripts/gcp_request_quota.sh <PROJECT_ID> us-central1 --dry-run   # inspect first
./scripts/gcp_request_quota.sh <PROJECT_ID> us-central1             # submit
```

It refuses to submit if billing is not enabled, since requests from a Free Trial account
are rejected outright.

**gcloud is installed** (`580.0.0` + `beta` component) but the homebrew cask does not put
it on your PATH. The script finds it either way; for interactive use add:

```bash
echo 'export PATH=/opt/homebrew/share/google-cloud-sdk/bin:"$PATH"' >> ~/.zshrc
```

Check whether the grant landed:

```bash
gcloud beta quotas preferences list --project=<PROJECT_ID>
gcloud compute regions describe us-central1 \
  --format="table(quotas.metric,quotas.limit,quotas.usage)" | grep -i l4
```

---

## 2. Cost guardrails — before you create anything

Upgrading to paid billing means a forgotten VM bills real money. Set these first.

**Budget alert.** Billing → Budgets & alerts → Create budget, scoped to the project, amount
$300, alerts at 50/90/100%. Note this only *notifies*; it does not stop spend.

**Auto-shutdown on idle.** The real protection. Baked into the startup script in step 3 —
it powers the box off after 30 minutes with no GPU utilisation.

**Watch the disk.** Persistent disk bills even while the VM is stopped (~$0.04/GB/month for
balanced). A 200GB disk left behind is ~$8/month indefinitely. Delete disks you are done
with; do not just stop the instance.

---

## 3. Create the L4 spot VM

L4 is only available on the **G2** machine family, where the GPU is part of the machine
type — there is no `--accelerator` flag, unlike T4 on N1.

| Machine type | vCPU | RAM | GPU |
|---|---|---|---|
| `g2-standard-4` | 4 | 16GB | 1× L4 (24GB) |
| `g2-standard-8` | 8 | 32GB | 1× L4 (24GB) |

Take `g2-standard-8` — dataloader workers for video-heavy VLA data will starve the GPU on
4 vCPU, and the extra vCPU is cheap relative to the GPU.

```bash
gcloud compute instances create vla-l4 \
  --zone=us-central1-a \
  --machine-type=g2-standard-8 \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --image-family=pytorch-latest-gpu --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB --boot-disk-type=pd-balanced \
  --metadata-from-file=startup-script=scripts/gcp_startup.sh,shutdown-script=scripts/gcp_shutdown.sh \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

Notes on the flags that matter:
- `--instance-termination-action=STOP` (not `DELETE`) keeps the boot disk on preemption, so
  a resumed run finds its checkpoints locally.
- The Deep Learning VM image ships CUDA and the NVIDIA driver already. Building from a bare
  Ubuntu image and installing drivers by hand wastes an hour of credit.
- `--scopes=cloud-platform` lets the box write to GCS without a key file.

Connect: `gcloud compute ssh vla-l4 --zone=us-central1-a`

---

## 4. Surviving preemption

Spot VMs get roughly **30 seconds** of notice, delivered as an ACPI shutdown that runs the
shutdown script. `common/trainer.py` already traps SIGTERM, finishes the current step, and
writes an atomic checkpoint — so the only thing needed is to route the signal and get the
checkpoint off the box.

That is what [`scripts/gcp_shutdown.sh`](../scripts/gcp_shutdown.sh) does: SIGTERM the
trainer, wait for it to checkpoint, then sync to GCS.

Create the bucket once:

```bash
gcloud storage buckets create gs://<BUCKET>-vla --location=us-central1
```

30 seconds is not much. Keep checkpoints small (LoRA adapters + optimizer state, not full
3B weights) or sync incrementally during training rather than only at shutdown.

---

## 5. Code and data sync

Code goes up by git; data and checkpoints go through GCS.

```bash
# on the VM
git clone <your-remote> ~/VLAProjects && cd ~/VLAProjects

# checkpoints down
gcloud storage rsync -r gs://<BUCKET>-vla/ckpt ./ckpt
```

For quick iteration without a push, `gcloud compute scp --recurse ./common vla-l4:~/VLAProjects/`
is fine — but anything you want to keep belongs in git or GCS, not on a spot VM's disk.

---

## 6. Verify the box before spending on it

```bash
nvidia-smi                       # expect: NVIDIA L4, 23034MiB
python -c "
import torch
from common.device import pick_device, describe
print(torch.__version__, torch.cuda.is_available())
print(describe(pick_device()))   # expect: NVIDIA L4 [cuda, bfloat16, 22.x GB]
"
```

L4 is Ada (sm_89), so `pick_device()` should resolve **bf16**, not fp16. If it reports
fp16 you are not on the GPU you think you are.

---

## 7. Stop the instance when you stop working

```bash
gcloud compute instances stop vla-l4 --zone=us-central1-a
```

A stopped instance bills only for disk. **Deleting** it (and its disk) is what actually
ends the spend:

```bash
gcloud compute instances delete vla-l4 --zone=us-central1-a
```

## Budget math

$300 against on-demand rates, before the vCPU/RAM/disk that rides along (budget another
$0.2–0.5/hr):

| GPU | Machine | On-demand | ≈ credit hours |
|---|---|---|---|
| T4 | N1 + accelerator | $0.54/hr | ~555h |
| **L4** | **g2-standard-8** | **$0.70/hr** | **~428h** |
| A100 40GB | a2-highgpu-1g | $3.67/hr | ~82h |

Spot cuts 60–91%, which puts L4 near $0.21–0.28/hr — comfortably over 1,000 GPU-hours for
the four-week plan. Use Kaggle for smoke tests so credits go to real training runs.
