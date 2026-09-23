# common — shared infrastructure

Three pieces every project here depends on, kept in one place because getting any of them
subtly wrong is invisible until a run is wasted.

| module | what it does |
|---|---|
| `device.py` | `pick_device()` — resolves device **and dtype from real compute capability**, never assumed. T4 is Turing (sm_75): fp16 only, no bf16. Hardcoding bf16 autocast fails there, and code must stay correct on both paths |
| `ckpt.py` | strict checkpoint loading that reports missing / unexpected / shape-mismatch separately. "It loaded" is not a result; **0/0/0** is |
| `trainer.py` | `ResumableTrainer` — checkpoints step-level state (model, optimizer, scheduler, sampler position, RNG, scaler) atomically and resumes cold |

`ResumableTrainer` exists because free-tier sessions are killed mid-run: Kaggle caps at
9 h, Colab at 12 h and neither guarantees availability. Every training run goes through
it rather than a bespoke loop, so a killed session costs the time since the last
checkpoint rather than the whole run.

No device is hardcoded anywhere in this repo — `pick_device()` is the single authority,
which is also what lets the same code path run CPU locally and CUDA remotely.
