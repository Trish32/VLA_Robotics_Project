#!/usr/bin/env python
"""Settle the kernel offset ORDER against MinkowskiEngine — without building it.

This is the check `minkowski_compat.py` could not make locally, and the one five Kaggle
GPU rounds failed to deliver. The breakthrough is that it never needed a GPU or a working
install: the enumeration lives in `src/kernel_region.hpp`, which is header-only template
code, and `-DCPU_ONLY` strips the `.cuh` includes.

So instead of building a CUDA extension that fails in a vendored nvtx header, this
compiles ~25 lines against ME's own header, asks `coordinate_at()` for every kernel index,
and diffs the result against `kernel_offsets()`. Upstream's own code is the oracle, and
it runs on a Mac in about two seconds.

Why the order could not be checked any other way: shapes are order-independent, so the
469-tensor Mask3D load passes under any permutation, and the dense-Conv3d oracle builds
its reference weight from the same offsets it is testing. A wrong order loads 0 missing /
0 unexpected and emits plausible garbage.

    python openmask3d_semantic/tools/verify_kernel_order.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ME_REPO = "https://github.com/NVIDIA/MinkowskiEngine.git"

# Three regimes, because the conventions differ across them:
#   odd kernel   -> offsets centred on 0
#   EVEN kernel  -> offsets start at 0, NOT centred (8 layers of Res16UNet34C)
#   stride > 1   -> offsets scale by the tensor stride
CASES = [(3, 1), (2, 1), (5, 1), (3, 8), (2, 2)]

DUMPER = """
#define CPU_ONLY
#include <cstdio>
#include <vector>
#include "kernel_region.hpp"
using namespace minkowski;
using ct = default_types::dcoordinate_type;
using st = default_types::size_type;
static void dump(st k, st tstride) {
  st D = 3;
  std::vector<st> ks(D, k), ts(D, tstride), dil(D, 1);
  cpu_kernel_region<ct> region(RegionType::HYPER_CUBE, D + 1, ts.data(),
                               ks.data(), dil.data());
  st volume = region.volume();
  std::vector<ct> src(D + 1, 0), dst(D + 1, 0);
  printf("k=%u stride=%u volume=%u\\n", (unsigned)k, (unsigned)tstride, (unsigned)volume);
  for (st i = 0; i < volume; ++i) {
    region.coordinate_at(i, src.data(), dst.data());
    printf("(%d,%d,%d)\\n", (int)dst[1], (int)dst[2], (int)dst[3]);
  }
}
int main() {
%CALLS%
  return 0;
}
"""


def _pybind_includes() -> list[str]:
    try:
        import pybind11
    except ImportError:
        raise SystemExit(
            "pybind11 is needed for its headers only (ME's types.hpp includes it):\n"
            "    pip install pybind11"
        ) from None
    return ["-I", pybind11.get_include(), "-I", sysconfig.get_paths()["include"]]


def build_and_run(workdir: Path) -> str:
    source = workdir / "ME"
    if not source.exists():
        print(f"[clone] {ME_REPO}")
        subprocess.run(["git", "clone", "-q", "--depth", "1", ME_REPO, str(source)],
                       check=True)

    robin = next(source.glob("src/**/robin_hood.h"), None)
    if robin is None:
        raise SystemExit("robin_hood.h not found; upstream layout changed")

    calls = "\n".join(f"  dump({k}, {s});" for k, s in CASES)
    cpp = workdir / "me_offsets.cpp"
    cpp.write_text(DUMPER.replace("%CALLS%", calls))

    binary = workdir / "me_offsets"
    cmd = ["c++", "-std=c++17", "-w", "-DCPU_ONLY",
           "-I", str(source / "src"), "-I", str(robin.parent),
           *_pybind_includes(), "-o", str(binary), str(cpp)]
    print("[build] compiling against ME's own kernel_region.hpp (CPU_ONLY, no CUDA)")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-2000:])
        raise SystemExit("compile failed")

    return subprocess.run([str(binary)], capture_output=True, text=True, check=True).stdout


def main() -> int:
    from openmask3d_semantic.minkowski_compat import kernel_offsets

    if shutil.which("c++") is None:
        raise SystemExit("no C++ compiler on PATH")

    with tempfile.TemporaryDirectory() as tmp:
        output = build_and_run(Path(tmp))

    blocks = re.split(r"k=(\d+) stride=(\d+) volume=(\d+)\n", output)[1:]
    all_ok = True
    print()
    print(f"{'case':<22}{'ME volume':>11}{'ours':>7}   verdict")
    print("-" * 56)
    for i in range(0, len(blocks), 4):
        k, stride, volume, body = (int(blocks[i]), int(blocks[i + 1]),
                                   int(blocks[i + 2]), blocks[i + 3])
        truth = [[int(v) for v in m]
                 for m in re.findall(r"\((-?\d+),(-?\d+),(-?\d+)\)", body)]
        mine = kernel_offsets(k, dimension=3, tensor_stride=stride).tolist()
        ok = truth == mine
        all_ok &= ok
        print(f"{f'k={k} stride={stride}':<22}{volume:>11}{len(mine):>7}   "
              f"{'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            for j, (a, b) in enumerate(zip(truth, mine)):
                if a != b:
                    print(f"    first difference at index {j}: ME {a} vs ours {b}")
                    break
    print("-" * 56)

    if all_ok:
        print("\nPASS — kernel offset order matches upstream exactly.")
        print("Axis 0 varies fastest; odd kernels centred, even kernels from 0;")
        print("offsets scale by the tensor stride. bug_log.txt [2] is closed.")
        return 0
    print("\nFAIL — our enumeration disagrees with upstream.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
