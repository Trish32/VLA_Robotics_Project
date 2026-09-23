#!/usr/bin/env python
"""Package the MinkowskiEngine diff as a Kaggle GPU kernel, push it, and wait.

Kaggle's free tier is the only T4 this project can reach, and the kernel runs in a fresh
container with no access to this repo — so the script it runs has to be self-contained.
`minkowski_compat.py` imports nothing but stdlib and torch, which makes that a
concatenation rather than a refactor: the module is inlined verbatim, then the comparison
is appended. Inlining verbatim matters — a trimmed copy would be testing something other
than what runs here.

The kernel reports honestly in both directions. MinkowskiEngine's last release predates
modern torch, so it may simply not build on Kaggle's CUDA-12 / torch-2.x image. That is
not a wasted run: it is direct evidence for the decision to replace it, and the kernel
records which install strategy was tried and how each failed.

    KAGGLE_API_TOKEN=... python openmask3d_semantic/tools/push_kaggle_verify.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "openmask3d_semantic" / ".kaggle_kernel"
SLUG = "minkowski-order-verify"

VERIFY_BODY = '''

# ===========================================================================
# Comparison against real MinkowskiEngine. Everything above is inlined verbatim
# from openmask3d_semantic/minkowski_compat.py.
# ===========================================================================
import subprocess, sys, traceback

TOL = 1e-4   # fp32 accumulation over up to 125 taps; a real ordering error is ~1e0


def install_minkowski():
    """Build MinkowskiEngine with BLAS specified explicitly.

    ROOT CAUSE, found by running setup.py directly on a Kaggle T4 (rounds 1-3 all hid it
    under pip's <pip-setuptools-caller> boilerplate):

        File "/tmp/ME/setup.py", line 201
            import numpy.distutils.system_info as sysinfo
        ModuleNotFoundError: No module named 'numpy.distutils'

    `numpy.distutils` was REMOVED in NumPy 2.0, and Kaggle ships NumPy 2.x. It is not
    stdlib distutils, which is what rounds 2 and 3 were aimed at.

    The fix is not a patch: that import sits in the `else` branch taken only when BLAS
    was not specified. `--blas=openblas` takes the other branch and never imports it.
    Modern pip dropped --install-option, so setup.py is invoked directly.
    """
    import os
    env = dict(os.environ, MAX_JOBS="2")

    def run(label, cmd, **kw):
        print(f"[install] {label} ...", flush=True)
        r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, **kw)
        stream = ((r.stdout or "") + (r.stderr or "")).strip()
        print(f"[install] {label} rc={r.returncode}")
        for line in stream.splitlines()[-30:]:
            print("   ", line)
        return r.returncode == 0

    run("openblas", "apt-get -qq update >/dev/null 2>&1 && "
        "apt-get -qq install -y libopenblas-dev build-essential >/dev/null 2>&1; echo ok")
    run("clone", "rm -rf /tmp/ME && git clone -q --depth 1 "
        "https://github.com/NVIDIA/MinkowskiEngine.git /tmp/ME && echo cloned")
    # Capture the WHOLE build log then print the FIRST error with context. Rounds 1-4
    # used `| tail -30`, which shows the last error -- and compiler errors cascade, so
    # the tail is the consequence and the head is the cause.
    run("build --blas=openblas", r"""
cd /tmp/ME && python setup.py install --blas=openblas > /tmp/build.log 2>&1
echo "exit=$?  lines=$(wc -l < /tmp/build.log)"
echo '--- first 3 errors with context ---'
grep -n -m3 -B4 -A12 -E 'error:|Error [0-9]|fatal error' /tmp/build.log || echo '(no error: lines)'
echo '--- last 8 lines ---'
tail -8 /tmp/build.log
""", env=env)

    try:
        import MinkowskiEngine  # noqa: F401
        return True
    except Exception as exc:
        print(f"[install] still not importable: {exc}")
        return False


def random_sparse(n_points=4000, channels=8, span=48, seed=0):
    rng = np.random.default_rng(seed)
    coords = np.unique(rng.integers(-span, span, size=(n_points, 3)), axis=0)
    batched = np.column_stack([np.zeros(len(coords), dtype=np.int64), coords])
    feats = rng.standard_normal((len(coords), channels)).astype(np.float32)
    return (torch.as_tensor(batched, dtype=torch.int32, device="cuda"),
            torch.as_tensor(feats, device="cuda"))


def compare(kernel_size, stride, c_in=8, c_out=6):
    import MinkowskiEngine as ME
    coords, feats = random_sparse(channels=c_in)

    me_conv = ME.MinkowskiConvolution(
        c_in, c_out, kernel_size=kernel_size, stride=stride, bias=False, dimension=3
    ).cuda()
    me_out = me_conv(ME.SparseTensor(features=feats, coordinates=coords))

    # The SAME weight tensor for both: this compares the OPERATION, not an init.
    weight = me_conv.kernel.detach().clone()
    print(f"    ME kernel shape {tuple(weight.shape)}")

    ours_in = SparseTensor(feats.cpu().double(), coords.cpu(), (1, 1, 1))
    ours_out = sparse_conv(ours_in, weight.cpu().double(), None, kernel_size, stride=stride)

    me_coords = me_out.C.cpu().numpy(); me_feats = me_out.F.detach().cpu().numpy()
    our_coords = ours_out.coordinates.numpy(); our_feats = ours_out.features.numpy()
    lookup = {tuple(r): i for i, r in enumerate(our_coords)}
    shared = [(i, lookup[tuple(r)]) for i, r in enumerate(me_coords) if tuple(r) in lookup]
    if not shared:
        return float("inf"), 0, len(me_coords), len(our_coords)

    mine = np.stack([our_feats[j] for _, j in shared])
    theirs = np.stack([me_feats[i] for i, _ in shared])
    return (float(np.abs(mine - theirs).max()), len(shared),
            len(me_coords), len(our_coords))


print("=" * 78)
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
print("=" * 78)

if not torch.cuda.is_available():
    print("RESULT: NO_GPU — kernel was not given an accelerator")
    raise SystemExit(0)

if not install_minkowski():
    print()
    print("RESULT: ME_UNBUILDABLE")
    print("MinkowskiEngine could not be built on this image. That is the very reason")
    print("minkowski_compat.py exists; record it in bug_log.txt [1] as supporting")
    print("evidence, and the kernel ORDER claim in [2] stays unverified.")
    raise SystemExit(0)

import MinkowskiEngine as ME
print("MinkowskiEngine", getattr(ME, "__version__", "?"))

cases = [
    ("k=3 s=1  odd, centered", 3, 1),
    ("k=2 s=2  EVEN, not centered", 2, 2),
    ("k=5 s=1  stem width", 5, 1),
    ("k=3 s=2  odd, strided", 3, 2),
]
worst = 0.0
print()
for label, k, s in cases:
    try:
        diff, shared, n_me, n_ours = compare(k, s)
    except Exception:
        traceback.print_exc()
        print(f"{label:<32} EXCEPTION")
        worst = float("inf")
        continue
    worst = max(worst, diff)
    verdict = "MATCH" if diff < TOL else "MISMATCH"
    print(f"{label:<32} coords ME={n_me} ours={n_ours} shared={shared} "
          f"max|diff|={diff:.3e}  {verdict}")

print()
if worst < TOL:
    print(f"RESULT: PASS  max deviation {worst:.3e} < {TOL:.0e}")
    print("Kernel offset ORDER and the correlation convention are confirmed.")
else:
    print(f"RESULT: FAIL  max deviation {worst:.3e}")
    print("Suspect, in order: (1) offset enumeration order (ME increments the FIRST")
    print("axis fastest), (2) even-kernel centering, (3) correlation sign.")
'''

METADATA = {
    "id": None,
    "title": SLUG,
    "code_file": "verify.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
}


def build(username: str) -> Path:
    STAGE.mkdir(parents=True, exist_ok=True)
    source = (ROOT / "openmask3d_semantic" / "minkowski_compat.py").read_text()
    (STAGE / "verify.py").write_text(source + VERIFY_BODY)

    metadata = dict(METADATA, id=f"{username}/{SLUG}")
    (STAGE / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    return STAGE


def main() -> int:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    # Token auth does not always populate config_values, so fall back to reading the
    # owner off any kernel the account already has.
    username = api.config_values.get("username")
    if not username:
        mine = api.kernels_list(mine=True, page_size=1)
        if mine:
            username = str(getattr(mine[0], "ref", "")).split("/")[0]
    if not username:
        raise SystemExit(
            "could not determine the Kaggle username. Set KAGGLE_USERNAME, or create "
            "one kernel in the web UI so the account has something to read it from."
        )

    staged = build(username)
    print(f"[push] {staged} -> {username}/{SLUG}")
    print(api.kernels_push(str(staged)))

    print("[wait] polling status (GPU kernels queue, this can take a few minutes)")
    for _ in range(90):
        time.sleep(20)
        status = api.kernels_status(f"{username}/{SLUG}")
        state = getattr(status, "status", status)
        print(f"   {state}", flush=True)
        if str(state).lower() in {"complete", "error", "cancelrequested", "cancelled"}:
            break

    out = STAGE / "output"
    out.mkdir(exist_ok=True)
    api.kernels_output(f"{username}/{SLUG}", str(out))
    print(f"[done] output in {out}")
    for path in sorted(out.glob("*.log")) + sorted(out.glob("*.txt")):
        print(f"--- {path.name}")
        print(path.read_text()[-4000:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
