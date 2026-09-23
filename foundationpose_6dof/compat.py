"""Let FoundationPose import on a Mac without faking anything it computes.

`Utils.py` imports `nvdiffrast.torch` and six `pytorch3d` symbols at module scope, and
`estimater.py` does `from Utils import *`. So on a box without those packages the whole
project is unimportable — including `learning/models/{score,refine}_network.py`, which
are plain PyTorch and have no CUDA dependency at all. That is the thing worth rescuing:
those two networks are the checkpoint, and the Fidelity Rule needs them loadable before
anything else happens.

CLAUDE.md prefers registry hooks over editing vendored source, so nothing here patches
upstream. Absent packages are registered in `sys.modules` as modules that import cleanly
and **raise on first attribute use** — the same contract
`droidSLAM_monocular/compat.py` gives `lietorch` and `droid_backends`, and for the same
reason: nvdiffrast rasterises the pose hypotheses that the refiner scores, so a stub
returning zeros would produce well-shaped poses that are confidently wrong.

The split, and why each side is where it is:

  * **nvdiffrast** — CUDA-only rasteriser, no CPU build. Stubbed, raises.
  * **pytorch3d** — genuinely has a CPU build, it is just painful to install on Apple
    Silicon. Stubbed only when absent, so a machine that has it uses the real thing.
  * **kornia / mycpp / bundlesdf.mycuda / warp** — already guarded by upstream itself
    (`Utils.py` wraps each in try/except), so they need nothing from us.

Nothing here makes FoundationPose *run* locally. It makes it *import* locally, which is
what separates "the checkpoint loads 0/0" from "we could not even try".
"""

from __future__ import annotations

import sys
import types
from typing import Any

# Submodules that `Utils.py` imports by name. A parent package alone is not enough:
# `from pytorch3d.transforms import so3_log_map` imports the submodule first.
PYTORCH3D_SUBMODULES = (
    "pytorch3d",
    "pytorch3d.transforms",
    "pytorch3d.renderer",
    "pytorch3d.renderer.mesh",
    "pytorch3d.renderer.mesh.rasterize_meshes",
    "pytorch3d.renderer.mesh.shader",
    "pytorch3d.renderer.mesh.textures",
    "pytorch3d.structures",
)

NVDIFFRAST_SUBMODULES = ("nvdiffrast", "nvdiffrast.torch")

NVDIFFRAST_WHY = (
    "nvdiffrast is a CUDA rasteriser with no CPU build. FoundationPose uses it to render "
    "every pose hypothesis before the refiner and scorer compare them to the observed "
    "crop, so it is the core of the algorithm rather than a utility."
)
PYTORCH3D_WHY = (
    "pytorch3d is not installed in this environment. It does have a CPU build, so this "
    "is a packaging gap rather than a hardware limit -- install it to lift the "
    "restriction."
)


class MissingModule(types.ModuleType):
    """Imports cleanly; raises with a useful message the moment it is touched.

    Subclasses `ModuleType` rather than being a bare object so that `import a.b.c` and
    `from a.b import name` both resolve through the normal machinery.
    """

    def __init__(self, name: str, why: str) -> None:
        super().__init__(name)
        self.__dict__["_why"] = why
        # A package needs __path__ for `import parent.child` to be attempted at all.
        self.__dict__["__path__"] = []

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)

        name, why = self.__name__, self.__dict__["_why"]

        def _fail(*_args: Any, **_kwargs: Any) -> Any:
            raise NotImplementedError(
                f"{name}.{attr} is unavailable here. {why}\n"
                "It is deliberately NOT stubbed with a working-looking fallback: a fake "
                "renderer returns plausible geometry and makes every pose silently "
                "wrong. Run this on the GPU host."
            )

        return _fail


def _register(names: tuple[str, ...], why: str) -> None:
    for name in names:
        if name in sys.modules:
            continue
        module = MissingModule(name, why)
        sys.modules[name] = module
        if "." in name:
            parent, _, leaf = name.rpartition(".")
            setattr(sys.modules[parent], leaf, module)


def _available(name: str) -> bool:
    """True only for the REAL package.

    The check has to exclude our own placeholders explicitly. Once a stub is in
    `sys.modules`, `__import__` succeeds on it, so a naive probe reports the package as
    present on every call after the first -- and `describe` would cheerfully announce
    that pose estimation can run on a machine with no renderer.
    """
    if isinstance(sys.modules.get(name), MissingModule):
        return False
    try:
        __import__(name)
    except Exception:
        return False
    return not isinstance(sys.modules.get(name), MissingModule)


def ensure_importable() -> dict[str, bool]:
    """Make `import Utils` succeed. Returns {package: True if real, False if stubbed}.

    Callers should log this: "the checkpoint loaded" means something different on a box
    where the renderer was a placeholder.
    """
    status: dict[str, bool] = {}

    for package, names, why in (
        ("nvdiffrast", NVDIFFRAST_SUBMODULES, NVDIFFRAST_WHY),
        ("pytorch3d", PYTORCH3D_SUBMODULES, PYTORCH3D_WHY),
    ):
        real = _available(package)
        status[package] = real
        if not real:
            _register(names, why)

    return status


def describe(status: dict[str, bool]) -> str:
    stubbed = sorted(name for name, real in status.items() if not real)
    if not stubbed:
        return "all render dependencies present; pose estimation can actually run"
    return (
        f"stubbed (raise-on-use): {', '.join(stubbed)} -- networks and checkpoints are "
        "usable, rendering and pose estimation are not"
    )
