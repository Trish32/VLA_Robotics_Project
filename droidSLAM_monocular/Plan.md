# DROID-SLAM — plan

**The binding gap: tracking has never run.** The checkpoint loads 0/0/0 and the
pure-PyTorch modules are unit-tested, but `lietorch` / `droid_backends` need a CUDA
compiler. A GPU box closes it; nothing else does.

Per the house rule, the extension must RAISE locally rather than be faked — a CPU stub
returning zeros passes shape tests and poisons every downstream result.

## Notes
- Covisibility precompute takes hours on first run and is cached — only relevant for training,
  which we are skipping.
- Expect this to be slower than the reference. A pure-PyTorch block-sparse Schur solve will not
  match hand-written CUDA; correctness first, then vectorise. Record the gap honestly.
- `MoyangLi00/DROID-W` (CVPR 2026) and `ChenHoy/DROID-Splat` are useful secondary references for
  how others have restructured this code.
