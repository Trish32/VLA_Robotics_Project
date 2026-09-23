"""`Gr00tPolicy` that runs on CPU/MPS.

Upstream's `Gr00tPolicy.__init__` hardcodes two things that make it CUDA-only:

    model = AutoModel.from_pretrained(model_dir)      # honours use_flash_attention=True
    model.to(device=device, dtype=torch.bfloat16)     # bf16 needs sm_80+ on GPU

The released config sets `use_flash_attention: true`, so `from_pretrained` builds the
flash path and dies on any machine without flash-attn. Rather than edit vendored source
we subclass and override construction, keeping every other method — observation
validation, processing, action decoding — exactly as upstream wrote it.

This is a **local correctness harness**, not a performance path. fp32 on CPU is slow and
the attention kernel differs from the reference; see bug_log [1] and [3]. Use it to prove
the plumbing works end to end before spending GPU time, not to produce a metric.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


class LocalGr00tPolicy:
    """Thin CPU/MPS-capable wrapper around upstream's Gr00tPolicy internals.

    Composes rather than inherits so that upstream's `__init__` never runs; every method
    it does define is delegated to the real class via `_delegate`.
    """

    def __init__(
        self,
        embodiment_tag,
        model_path: str | Path,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        strict: bool = True,
    ):
        import gr00t.model  # noqa: F401  (registers the model with transformers)
        from transformers import AutoProcessor

        from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
        from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        model_dir = Path(model_path)
        cfg = Gr00tN1d6Config(**json.loads((model_dir / "config.json").read_text()))
        # The two substitutions. Neither renames or reshapes a parameter, so the
        # checkpoint still loads 0 missing / 0 unexpected (tools/load_checkpoint.py).
        cfg.use_flash_attention = False
        cfg.load_bf16 = False
        cfg.model_dtype = str(dtype).replace("torch.", "")
        cfg.torch_dtype = dtype

        model = Gr00tN1d6.from_pretrained(model_dir, config=cfg)
        model.eval().to(device=device, dtype=dtype)

        self.model = model
        self.device = device
        self.dtype = dtype
        self.strict = strict
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        self.processor.eval()
        self.embodiment_tag = embodiment_tag
        self.modality_configs = self.processor.get_modality_configs()[embodiment_tag.value]
        self.collate_fn = self.processor.collator

        language_keys = self.modality_configs["language"].modality_keys
        assert len(language_keys) == 1, "Only one language key is supported"
        self.language_key = language_keys[0]

        self._delegate = Gr00tPolicy

    def __getattr__(self, name):
        """Fall through to upstream's implementation, bound to this instance.

        Keeps observation checking / action decoding identical to upstream instead of
        restating it here, where it would drift.
        """
        attr = getattr(self._delegate, name, None)
        if attr is None or not callable(attr):
            raise AttributeError(name)
        return attr.__get__(self, type(self))

    def get_action(self, observation: dict, options: dict | None = None):
        """Matches `BasePolicy.get_action(observation, options=None)`.

        `options` must be accepted even though it is unused here: upstream's PolicyServer
        dispatches with `handler(**data)` and its client always sends an `options` key,
        so dropping the parameter breaks every networked call while direct calls keep
        working.
        """
        return self._delegate.get_action(self, observation, options)
