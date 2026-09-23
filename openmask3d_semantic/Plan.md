# OpenMask3D — plan

**The binding gap: nothing here measures segmentation *accuracy*.**

| item | blocker | what closes it |
|---|---|---|
| **mIoU / open-vocab recall** | ScanNet200 is a gated ~1 TB download | ground-truth instance labels on any scene the stack can fuse |
| **MobileSAM precision cost** | needs the above | `vit_t` is the one acceleration step that changes outputs, so it must be measured against `vit_h` on the same scene rather than assumed free |
| **Wall-clock speedup, in seconds** | modelled only | a CUDA box with the real weights. Until then the acceleration numbers stay **MODELLED** and are never reported in seconds |

## Recognition is weak, measured on real scenes

Median top-1 CLIP margin over the pipeline's instances is **0.011**; the worst case
separates `table` from `desk` by **0.001**. At that separation an argmax over a fixed
vocabulary is close to arbitrary.

Prompt-template ensembling was tried and **did not help**: it raised absolute
similarities (0.198–0.239 → 0.21–0.28) and flipped one instance from `monitor` to
`person`. Better numbers, no better decisions — recorded because it reads as progress.

The fix that did land is architectural, not numerical: grounding ranks the saved
per-instance CLIP features against the query rather than string-matching a lossy argmax,
so the fixed vocabulary is out of the critical path (`pipeline/tools/e2e_ground.py`).

## Loose end

`pipeline/observations.to_scene_nodes` decides `SURFACE` vs `OBJECT` from a hardcoded
six-word set inside an open-vocabulary pipeline, and it already disagrees with the CLIP
vocabulary it is fed. See the root [Plan.md](../Plan.md) §5.
