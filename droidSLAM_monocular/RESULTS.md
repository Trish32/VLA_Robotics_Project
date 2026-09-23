# DROID-SLAM — results

What is verified. **Tracking has never executed**: `lietorch` and `droid_backends` are CUDA-compile-only and will not build on Apple
Silicon, so every SLAM number in this repo comes from ORB-SLAM3. See [Plan.md](Plan.md).

## Validation

- [x] **`droid.pth` loads 0 missing / 0 unexpected / 0 shape mismatch** — 102 tensors,
      4.00M params, `strict=True` succeeds on CPU →
      [tools/load_checkpoint.py](tools/load_checkpoint.py). **Fidelity gate passed.**
      One wrinkle, worth knowing: the checkpoint's `update.{delta,weight}.2` heads have
      **3 output channels** while the network declares 2. That is not a mismatch to fix —
      `droid.py::load_weights` keeps the first two channels and discards the third, so our
      loader replicates that slice. See `bug_log.txt` [2], including the wrong conclusion I
      recorded before reading upstream's loader.
- [ ] Per-op diff against the reference on identical saved inputs: correlation volume, then a
      single BA iteration, then the full frontend. Do not skip to end-to-end.
- [ ] ATE on **TUM-RGBD** (monocular) and **EuRoC** (mono + stereo) vs the paper. TUM-RGBD is the
      cheap one to start with — small sequences, fast turnaround.
- [ ] ETH3D (RGB-D) if time allows.

