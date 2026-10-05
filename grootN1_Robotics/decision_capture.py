"""Lossless, passive GR00T decision capture and same-input stochastic replay.

The hook reads the first upstream action-encoder input (flow-matching noise).
It does not replace noise, change policy code or consume random draws.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture_rng(device):
    import torch
    device = torch.device(device)
    name, keys, pos, has_gauss, cached_gauss = np.random.get_state()
    result = {"python": random.getstate(), "numpy": {
        "name": name, "keys": torch.tensor(keys.astype(np.int64)),
        "pos": pos, "has_gauss": has_gauss, "cached_gauss": cached_gauss},
        "torch_cpu": torch.get_rng_state().clone(), "device": str(device)}
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        result["device"] = f"cuda:{index}"
        result["torch_device"] = torch.cuda.get_rng_state(index).clone()
    elif device.type == "mps":
        result["torch_device"] = torch.mps.get_rng_state().clone()
    elif device.type != "cpu":
        raise ValueError(f"unsupported RNG device: {device}")
    return result


def restore_rng(state):
    import torch
    random.setstate(state["python"])
    saved = state["numpy"]
    np.random.set_state((saved["name"], saved["keys"].numpy().astype(np.uint32),
                         saved["pos"], saved["has_gauss"], saved["cached_gauss"]))
    torch.set_rng_state(state["torch_cpu"])
    device = torch.device(state["device"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_device"], device)
    elif device.type == "mps":
        torch.mps.set_rng_state(state["torch_device"])


def rng_equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(rng_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(rng_equal(x, y) for x, y in zip(a, b))
    return a == b


def seed_rng(seed, device):
    """Seed only the policy backend; leave other CUDA generators alone."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.init()  # default_generators is empty before lazy CUDA initialization.
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.default_generators[index].manual_seed(seed)
    elif device.type == "mps":
        torch.mps.manual_seed(seed)


@contextmanager
def isolated_rng(device):
    current = capture_rng(device)
    try:
        yield
    finally:
        restore_rng(current)


@contextmanager
def initial_noise(policy):
    """Use the pinned upstream encoder boundary, failing if it is not exercised."""
    model = policy.policy.model  # official Gr00tSimPolicyWrapper
    captured = []
    def read_noise(module, args):
        if not captured:
            noise, timesteps = args[:2]
            if not (timesteps == 0).all().item():
                raise ValueError("first action-encoder call is not initial flow noise")
            captured.append(noise.detach().cpu().clone())
    handle = model.action_head.action_encoder.register_forward_pre_hook(read_noise)
    try:
        yield captured
    finally:
        handle.remove()


def array_copy(values):
    arrays = {key: np.array(value, copy=True) for key, value in values.items()}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("object arrays are not lossless policy inputs")
    return arrays


class DecisionCapture:
    def __init__(self, directory, steps, device, provenance, reader=None):
        self.directory = Path(directory)
        self.steps = frozenset(steps)
        if not self.steps or min(self.steps) < 0:
            raise ValueError("capture steps must be nonnegative and nonempty")
        self.device, self.provenance, self.reader = device, provenance, reader
        self.records = []

    def predict(self, policy, observation, *, seed, sim_step, env):
        if sim_step not in self.steps:
            started = time.perf_counter()
            result = policy.get_action(observation)
            self.inference_seconds = time.perf_counter() - started
            return result
        import torch
        target = self.directory / f"episode_{seed}" / f"step_{sim_step}"
        target.mkdir(parents=True, exist_ok=False)
        inputs = array_copy(observation)
        labels = self.reader(env) if self.reader is not None else None
        before = capture_rng(self.device)
        with initial_noise(policy) as noise:
            started = time.perf_counter()
            actions, info = policy.get_action(observation)
            self.inference_seconds = time.perf_counter() - started
        after = capture_rng(self.device)
        if len(noise) != 1:
            raise ValueError("upstream initial flow noise was not captured")
        # Save only safe tensor/primitive data; never pickle a policy or simulator.
        np.savez_compressed(target / "observation.npz", **inputs)
        np.savez_compressed(target / "actions.npz", **array_copy(actions))
        torch.save({"before": before, "after": after, "initial_noise": noise[0]}, target / "rng.pt")
        manifest = {"format_version": 1, "seed": seed, "sim_step": sim_step,
                    "provenance": self.provenance, "labels": labels,
                    "observation_containers": {key: "list" if isinstance(value, list)
                        else "tuple" if isinstance(value, tuple) else "ndarray"
                        for key, value in observation.items()},
                    "files": {name: file_hash(target / name) for name in
                              ["observation.npz", "actions.npz", "rng.pt"]}}
        temporary = target / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        temporary.replace(target / "manifest.json")
        self.records.append(str(target.relative_to(self.directory)))
        return actions, info


def load_capture(directory):
    import torch
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    names = {"observation.npz", "actions.npz", "rng.pt"}
    if manifest.get("format_version") != 1 or set(manifest["files"]) != names:
        raise ValueError("unsupported or incomplete decision capture")
    for name, expected in manifest["files"].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"decision capture hash differs: {name}")
    def read(name):
        with np.load(directory / name, allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}
    observation = read("observation.npz")
    containers = manifest["observation_containers"]
    if set(containers) != set(observation) or any(
            kind not in {"ndarray", "list", "tuple"} for kind in containers.values()):
        raise ValueError("invalid observation container schema")
    for key, kind in containers.items():
        if kind == "list":
            observation[key] = observation[key].tolist()
        elif kind == "tuple":
            observation[key] = tuple(observation[key].tolist())
    return manifest, observation, read("actions.npz"), torch.load(
        directory / "rng.pt", map_location="cpu", weights_only=True)


def replay_decision(policy, directory, device, candidates=0, candidate_dir=None):
    """Verify exact reference replay, then sample alternatives without affecting RNG.

    Alternatives measure prediction variation only. No simulator branch is run,
    so this function never assigns candidate quality or claims selection headroom.
    """
    import torch
    if candidates < 0:
        raise ValueError("candidate count must be nonnegative")
    manifest, observation, expected, saved = load_capture(directory)
    if candidate_dir is not None:
        candidate_dir = Path(candidate_dir)
        candidate_dir.mkdir(parents=True, exist_ok=False)
    actual_device = capture_rng(device)["device"]
    if actual_device != saved["before"]["device"]:
        raise ValueError("exact replay requires the recorded device backend/index")
    with isolated_rng(device):
        policy.reset()
        restore_rng(saved["before"])
        with initial_noise(policy) as noise:
            actions, _ = policy.get_action(observation)
        if (len(noise) != 1 or not torch.equal(noise[0], saved["initial_noise"])
                or not rng_equal(capture_rng(device), saved["after"])):
            raise ValueError("replayed noise or post-call RNG differs")
        if set(actions) != set(expected) or any(
            np.asarray(actions[k]).dtype != expected[k].dtype or
            np.asarray(actions[k]).shape != expected[k].shape or
            not np.array_equal(actions[k], expected[k]) for k in expected):
            raise ValueError("replayed action chunk differs")
        differences = []
        for index in range(candidates):
            # Independent, stable seeds from capture identity; no search for a good draw.
            identity = f"candidate-v1:{manifest['seed']}:{manifest['sim_step']}:{index}"
            seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big")
            seed_rng(seed, device)
            policy.reset()
            candidate, _ = policy.get_action(observation)
            if set(candidate) != set(expected):
                raise ValueError("candidate action keys differ")
            groups = {}
            for key, value in candidate.items():
                value = np.asarray(value)
                if value.shape != expected[key].shape or not np.isfinite(value).all():
                    raise ValueError(f"invalid candidate action: {key}")
                delta = value.astype(np.float64) - expected[key]
                groups[key] = {"rms_difference": float(np.sqrt(np.mean(delta**2))),
                               "max_abs_difference": float(np.abs(delta).max())}
            row = {"seed": seed, "groups": groups}
            if candidate_dir is not None:
                path = candidate_dir / f"candidate_{index}.npz"
                np.savez_compressed(path, **array_copy(candidate))
                row["actions_file"] = path.name
                row["actions_sha256"] = file_hash(path)
            differences.append(row)
    return {"seed": manifest["seed"], "sim_step": manifest["sim_step"],
            "exact_replay_verified": True, "labels": manifest["labels"],
            "candidate_quality_evaluated": False, "candidate_differences": differences}
