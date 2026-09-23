# OpenMask3D — results

Full detail, including the acceleration analysis and the MinkowskiEngine build attempts,
is in the root [RESULTS.md](../RESULTS.md) §2. Summarised here.

## Checkpoint fidelity · MEASURED

| model | tensors | params | strict load |
|---|---|---|---|
| Mask3D (full) | 469 | 39.66 M | **0 missing / 0 unexpected / 0 shape-mismatch** |
| └ Res16UNet34C backbone | 374 | 37.87 M | 0 / 0 / 0 |

Loaded on CPU through the pure-PyTorch replacement, upstream byte-identical to `3bc3fc5`.

## The sparse convolution, verified two ways · MEASURED

**Dense equivalence.** On a fully occupied grid a sparse convolution has no sparsity left
to exploit and must equal `nn.Conv3d`. Agreement to **atol 1e-10** at k=3 and k=5.

**Kernel offset order.** The dense oracle *cannot* catch an ordering error — it builds its
reference weight from the same offsets under test, so any self-consistent order passes,
the checkpoint still loads 0/0/0, and the network emits plausible garbage. So the
enumeration is diffed against MinkowskiEngine's own `src/kernel_region.hpp`, compiled
`-DCPU_ONLY` (no CUDA, no install): **195 offsets, exact agreement** across k=3/k=5,
even kernels, and stride scaling.

Three ME conventions are load-bearing and all invisible to a shape check: axis 0 varies
fastest, odd kernels centre while **even kernels start at 0**, and offsets scale by the
tensor stride. Eight layers of Res16UNet34C use k=2.

Reproduce: `tools/verify_kernel_order.py`.

## Tests

39, CPU only. No CUDA extension is built or required.
