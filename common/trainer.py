"""Resumable training for free-tier GPU sessions.

Kaggle caps sessions at 9h, Colab at 12h, and both preempt without warning. A run
that only checkpoints per-epoch loses hours; a run that checkpoints weights but not
optimizer/RNG/data position resumes into a different trajectory and quietly
invalidates the experiment. So state here is step-level and complete.

Design:
  - checkpoints are atomic (write tmp, fsync, rename) so a kill mid-write cannot
    corrupt the file you are about to resume from;
  - `time_budget_s` stops the run cleanly before the platform kills it, leaving a
    final checkpoint rather than a truncated one;
  - SIGTERM/SIGINT flip the same flag, so preemption degrades to a clean stop;
  - the data position is stored as (epoch, step_in_epoch) and replayed by the
    dataset side via `set_epoch`/skip, since free-tier dataloaders are not seekable.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import signal
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn as nn

from .device import Accelerator


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    step_in_epoch: int = 0
    best_metric: float = math.inf
    elapsed_s: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)


class ResumableTrainer:
    """Minimal training loop with complete, atomic, step-level checkpointing.

    Args:
        model, optimizer: standard.
        acc: accelerator from `common.device.pick_device()`; supplies autocast
            dtype and the fp16 grad scaler.
        out_dir: checkpoint directory. Resume is automatic if `last.pt` exists.
        accum_steps: gradient accumulation, the usual way to fake batch size on 16GB.
        time_budget_s: stop cleanly after this much wall clock. Set below the
            platform session cap (e.g. 8.5*3600 for Kaggle's 9h).
        ckpt_every: checkpoint cadence in optimizer steps.
        grad_clip: max grad norm, or None.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        acc: Accelerator,
        out_dir: str | Path,
        *,
        scheduler: Any = None,
        accum_steps: int = 1,
        time_budget_s: float | None = None,
        ckpt_every: int = 500,
        keep_last: int = 2,
        grad_clip: float | None = 1.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.acc = acc
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.accum_steps = accum_steps
        self.time_budget_s = time_budget_s
        self.ckpt_every = ckpt_every
        self.keep_last = keep_last
        self.grad_clip = grad_clip

        self.scaler = acc.grad_scaler()
        self.state = TrainState()
        self._stop = False
        self._t0 = time.time()

        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(ValueError):  # not in main thread
                signal.signal(sig, self._on_signal)

    # ---------------------------------------------------------------- resume

    def maybe_resume(self, path: str | Path | None = None) -> bool:
        """Restore from `last.pt` (or `path`). Returns True if a run was resumed."""
        ckpt_path = Path(path) if path else self.out_dir / "last.pt"
        if not ckpt_path.exists():
            return False

        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(blob["model"])
        self.optimizer.load_state_dict(blob["optimizer"])
        if self.scheduler is not None and blob.get("scheduler") is not None:
            self.scheduler.load_state_dict(blob["scheduler"])
        if self.scaler is not None and blob.get("scaler") is not None:
            self.scaler.load_state_dict(blob["scaler"])
        self.state = TrainState(**blob["state"])
        _set_rng(blob["rng"])
        print(
            f"[resume] step {self.state.step}, epoch {self.state.epoch}, "
            f"{self.state.elapsed_s / 3600:.2f}h already spent"
        )
        return True

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        loader_fn: Callable[[int], Iterable],
        loss_fn: Callable[[nn.Module, Any], torch.Tensor | tuple[torch.Tensor, dict]],
        max_epochs: int,
        *,
        on_epoch_end: Callable[[int], dict[str, float]] | None = None,
        log_every: int = 50,
    ) -> TrainState:
        """Train until `max_epochs`, the time budget, or a stop signal.

        `loader_fn(epoch)` must build a fresh loader for that epoch (this is where
        you set the sampler seed so resumption is reproducible). `loss_fn(model,
        batch)` returns the loss, optionally with a dict of scalars to log.
        """
        self.model.train()
        while self.state.epoch < max_epochs and not self._should_stop():
            loader = loader_fn(self.state.epoch)
            skip = self.state.step_in_epoch  # replay position after a resume
            if skip:
                print(f"[resume] skipping {skip} batches into epoch {self.state.epoch}")

            for i, batch in enumerate(loader):
                if i < skip:
                    continue
                if self._should_stop():
                    break

                metrics = self._train_batch(loss_fn, batch, i)
                self.state.step_in_epoch = i + 1

                if metrics and self.state.step % log_every == 0:
                    self._log(metrics)
                if self.state.step and self.state.step % self.ckpt_every == 0:
                    self.save("last.pt")

            if not self._should_stop():
                self.state.epoch += 1
                self.state.step_in_epoch = 0
                if on_epoch_end is not None:
                    val = on_epoch_end(self.state.epoch)
                    self._log(val, prefix="val")
                    primary = next(iter(val.values()), None)
                    if primary is not None and primary < self.state.best_metric:
                        self.state.best_metric = primary
                        self.save("best.pt")
                self.save("last.pt")

        self.save("last.pt")
        reason = "stop signal / time budget" if self._should_stop() else "max_epochs"
        print(f"[done] {reason}; step {self.state.step}, {self._elapsed()/3600:.2f}h")
        return self.state

    def _train_batch(self, loss_fn, batch, i: int) -> dict[str, float]:
        with self.acc.autocast():
            out = loss_fn(self.model, batch)
        loss, extra = out if isinstance(out, tuple) else (out, {})
        scaled = loss / self.accum_steps

        if self.scaler is not None:
            self.scaler.scale(scaled).backward()
        else:
            scaled.backward()

        if (i + 1) % self.accum_steps != 0:
            return {}

        if self.grad_clip is not None:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()

        self.state.step += 1
        return {"loss": float(loss.detach()), **{k: float(v) for k, v in extra.items()}}

    # ------------------------------------------------------------ checkpoint

    def save(self, name: str = "last.pt") -> Path:
        """Atomically write full training state."""
        self.state.elapsed_s = self._elapsed()
        blob = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "state": asdict(self.state),
            "rng": _get_rng(),
        }
        final = self.out_dir / name
        tmp = final.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            torch.save(blob, f)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(final)  # atomic on POSIX

        if name == "last.pt":
            snap = self.out_dir / f"step_{self.state.step:08d}.pt"
            if self.ckpt_every and self.state.step % self.ckpt_every == 0:
                snap.write_bytes(final.read_bytes())
                self._prune()
        return final

    def _prune(self) -> None:
        snaps = sorted(self.out_dir.glob("step_*.pt"))
        for old in snaps[: -self.keep_last] if self.keep_last else []:
            old.unlink(missing_ok=True)

    # ----------------------------------------------------------------- misc

    def _on_signal(self, signum, frame) -> None:  # noqa: ARG002
        print(f"\n[signal {signum}] finishing current step, then checkpointing.")
        self._stop = True

    def _should_stop(self) -> bool:
        if self._stop:
            return True
        if self.time_budget_s is not None and self._elapsed() >= self.time_budget_s:
            self._stop = True
        return self._stop

    def _elapsed(self) -> float:
        return self.state.elapsed_s + (time.time() - self._t0)

    def _log(self, metrics: dict[str, float], prefix: str = "train") -> None:
        row = {"step": self.state.step, "epoch": self.state.epoch, **metrics}
        self.state.history.append({"split": prefix, **row})
        body = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in metrics.items())
        print(f"[{prefix}] step {self.state.step} {body}")
        (self.out_dir / "history.jsonl").open("a").write(
            json.dumps({"split": prefix, **row}) + "\n"
        )


def _get_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng(rng: dict[str, Any]) -> None:
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
