#!/usr/bin/env python
"""Push the rebuilt bundle and re-run `register()` on a Kaggle T4, then pull the result.

Three steps that were previously done by hand, which is how a bundle and a result drifted
apart in the first place (`bug_log.txt` [8]):

  1. **preflight** — the kernel's own bundle path, run locally. Refuses before a session
     is spent if a key the kernel reads is missing or the depth clamp eats the mask.
  2. **dataset version** — the bundle has changed, so `trishli/foundationpose-bundle`
     gets a new version. The kernel reads the latest, so skipping this step silently
     re-runs the OLD input, which is the exact failure the fingerprint exists to catch.
  3. **kernel push, poll, pull** — the result lands in `pose_bundle/pose_result.json`,
     carrying `bundle_fingerprint`, so `pipeline/tools/e2e_pose.py` can tell whether what
     it is reading was computed from the bundle on disk.

Credentials come from the environment — `KAGGLE_API_TOKEN`, or `KAGGLE_USERNAME` plus
`KAGGLE_KEY`. Nothing here reads or writes `~/.kaggle/`, and nothing prints a token.

    export KAGGLE_API_TOKEN=...        # in your own shell
    conda run -n foundationpose_vl python foundationpose_6dof/tools/push_kaggle_pose.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = Path(os.environ.get("E2E_DIR", ROOT / "pipeline/assets/e2e")) / "pose_bundle"
KAGGLE = ROOT / "foundationpose_6dof/.kaggle"
SLUG = "foundationpose-nvdiffrast"
DATASET = "foundationpose-bundle"

#: Files the kernel reads. Anything else in the bundle directory is ours, not Kaggle's.
SHIPPED = ("bundle.json", "mesh.obj", "dataset-metadata.json")


def authenticated_api():
    """The Kaggle client, authenticated — or None, with the reason printed.

    This asks the client whether it can authenticate rather than guessing at HOW. An
    earlier version checked for `KAGGLE_API_TOKEN` in the environment and refused when it
    was absent, which was wrong on this machine: kaggle 2.2.4 accepts an OAuth cache
    (`kaggle auth login`), a bare token at ~/.kaggle/access_token, AND the env var. The
    env check refused a session that was perfectly able to authenticate. Reimplementing a
    client's credential resolution is a way to disagree with it.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as exc:                       # the client prints its own setup help
        print(f"[push]  Kaggle authentication failed: {type(exc).__name__}\n"
              f"        {str(exc).splitlines()[0] if str(exc) else ''}\n"
              f"        Any one of these works: `kaggle auth login` (OAuth, no stored\n"
              f"        secret), a token at ~/.kaggle/access_token, or KAGGLE_API_TOKEN\n"
              f"        exported from a sourced profile. This file never prints a token.")
        return None




def ourway_source() -> str:
    """The exact source of the chair's mesh recipe, for the job to exec.

    `reject_outliers` and `poisson_mesh` are read with inspect from the modules the chair
    bundle was built with, and `mesh_control.py` is read whole. Shipping source rather
    than re-typing it in the kernel is the point: the control asks whether OUR mesh
    pipeline is the limit, and a re-implementation would answer a different question.
    The result is exec'd once here first, so a missing name fails locally, not on a T4.
    """
    import inspect

    sys.path.insert(0, str(ROOT))
    from pipeline.observations import reject_outliers
    from pipeline.tools.build_pose_bundle import poisson_mesh

    module = (ROOT / "foundationpose_6dof/mesh_control.py").read_text()
    module = "\n".join(l for l in module.splitlines()
                       if not l.startswith("from __future__"))
    src = ("import numpy as np\n\n" + inspect.getsource(reject_outliers) + "\n\n"
           + inspect.getsource(poisson_mesh) + "\n\n" + module)
    ns: dict = {}
    exec(src, ns)
    for name in ("reject_outliers", "poisson_mesh", "fuse_object", "ourway_mesh",
                 "view_angles", "rotation_deg", "rotation_deg_mod_flip", "relative",
                 "view_span", "track_depth_residual", "depth_agreement"):
        if name not in ns:
            raise SystemExit(f"ourway source does not define {name}")
    return src


def target_bundles() -> dict[str, Path]:
    """Extra bundles to pose in the same job, from TARGET_BUNDLES (colon-separated dirs).

    Each is a directory build_sam_bundle.py wrote; it ships under targets/<name>/.
    """
    out = {}
    for d in filter(None, os.environ.get("TARGET_BUNDLES", "").split(":")):
        meta = json.load(open(Path(d) / "bundle.json"))
        out[meta["target"]] = Path(d)
    return out


def wait_for_dataset(api, ref: str, fingerprint: str, minutes: float = 15.0,
                     manifest: dict | None = None) -> bool:
    """Block until Kaggle SERVES the bundle we just uploaded, not merely accepts it.

    `dataset_create_version` returns as soon as the upload is accepted; the version is
    then processed asynchronously, and a kernel pushed in that window mounts the LAST
    READY version. That is what happened on the first v15 run: the upload landed, the
    kernel started seconds later, and it mounted the previous bundle (fingerprint None,
    695 px) — caught only because the kernel checks EXPECTED_FINGERPRINT.

    So readiness is checked the only way that cannot be fooled: download bundle.json
    from the live dataset and compare its fingerprint to the one we pushed.
    """
    import tempfile
    import zipfile

    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        try:
            status = str(api.dataset_status(ref)).split(".")[-1].lower()
        except Exception as exc:                       # transient API errors: retry
            status = f"unknown ({type(exc).__name__})"
        live, live_manifest = None, None
        if status == "ready":
            with tempfile.TemporaryDirectory() as tmp:
                api.dataset_download_file(ref, "bundle.json", path=tmp, quiet=True)
                if manifest is not None:
                    api.dataset_download_file(ref, "targets.json", path=tmp, quiet=True)
                for z in Path(tmp).glob("*.zip"):
                    zipfile.ZipFile(z).extractall(tmp)
                f = Path(tmp) / "bundle.json"
                live = json.load(open(f)).get("fingerprint") if f.exists() else None
                g = Path(tmp) / "targets.json"
                live_manifest = json.load(open(g)) if g.exists() else None
        print(f"   dataset {status}, serving fingerprint {live}"
              + (f", targets {live_manifest}" if manifest is not None else ""), flush=True)
        # With targets, the root bundle alone proves nothing: it is often UNCHANGED from
        # the previous version, so it reads as current before the new version has landed
        # — the [10] race again. The manifest is the file that changes.
        if live == fingerprint and (manifest is None or live_manifest == manifest):
            return True
        time.sleep(20)
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-dataset", action="store_true",
                    help="kernel only; the bundle on Kaggle is already current")
    ap.add_argument("--poll-minutes", type=float, default=30.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="preflight and stage the upload, then stop before authenticating "
                         "— so the handoff can be verified without a token")
    a = ap.parse_args(argv)

    api = None
    if not a.dry_run:
        api = authenticated_api()
        if api is None:
            return 2

    # 1. preflight -------------------------------------------------------------
    pre = subprocess.run([sys.executable, str(ROOT / "foundationpose_6dof/tools/preflight_bundle.py")],
                         cwd=str(ROOT))
    if pre.returncode:
        print("[push]  preflight refused the bundle; not spending a session on it")
        return 1

    meta = json.load(open(BUNDLE / "bundle.json"))
    print(f"[push]  bundle fingerprint {meta['fingerprint']}, "
          f"anchor {meta['frames'][0]['mask_pixels']} px")

    if a.dry_run:
        username = os.environ.get("KAGGLE_USERNAME", "trishli")
    else:
        # Token/OAuth auth does not always populate config_values, so fall back to
        # reading the owner off a kernel the account already has.
        username = api.config_values.get("username")
        if not username:
            mine = api.kernels_list(mine=True, page_size=1)
            username = str(getattr(mine[0], "ref", "")).split("/")[0] if mine else None
        if not username:
            raise SystemExit("could not determine the Kaggle username; set KAGGLE_USERNAME")
        print(f"[push]  authenticated as {username}")

    # 2. dataset version -------------------------------------------------------
    if not a.skip_dataset:
        stage = ROOT / "foundationpose_6dof/.kaggle/_dataset"
        shutil.rmtree(stage, ignore_errors=True)
        stage.mkdir(parents=True)
        for name in SHIPPED:
            shutil.copy2(BUNDLE / name, stage / name)
        for rec in meta["frames"]:
            i = rec["frame"]
            for name in (f"rgb_{i:03d}.png", f"depth_{i:03d}.png", f"mask_{i:03d}.png"):
                shutil.copy2(BUNDLE / name, stage / name)
        targets = target_bundles()
        if targets:
            for name, d in targets.items():
                dst = stage / "targets" / name
                dst.mkdir(parents=True)
                for f in d.iterdir():
                    if f.name != "dataset-metadata.json" and f.is_file():
                        shutil.copy2(f, dst / f.name)
            (stage / "targets.json").write_text(json.dumps(
                {n: json.load(open(d / "bundle.json"))["fingerprint"]
                 for n, d in sorted(targets.items())}, sort_keys=True))
            print(f"[push]  + {len(targets)} target bundles: {sorted(targets)}")
        (stage / "dataset-metadata.json").write_text(json.dumps(
            {"title": DATASET, "id": f"{username}/{DATASET}",
             "licenses": [{"name": "CC0-1.0"}]}, indent=1))
        total = sum(f.stat().st_size for f in stage.iterdir()) / 1e6
        print(f"[push]  dataset {username}/{DATASET}: {len(list(stage.iterdir()))} files, "
              f"{total:.1f} MB")
        if not a.dry_run:
            api.dataset_create_version(
                str(stage), version_notes=f"bundle {meta['fingerprint']}: "
                f"visibility-selected frames, dense depth-verified masks")

    if not a.dry_run:
        # Whether or not this run uploaded, never push the kernel until Kaggle is
        # serving the bundle on disk — otherwise the job mounts whatever was last ready.
        print(f"[push]  waiting for {username}/{DATASET} to serve {meta['fingerprint']}")
        targets = target_bundles()
        manifest = ({n: json.load(open(d / "bundle.json"))["fingerprint"]
                     for n, d in sorted(targets.items())} if targets else None)
        if not wait_for_dataset(api, f"{username}/{DATASET}", meta["fingerprint"],
                                manifest=manifest):
            print("[push]  the dataset never served this bundle; not pushing a kernel "
                  "that would run on a different input")
            return 1

    # 3. kernel ----------------------------------------------------------------
    stage_k = ROOT / "foundationpose_6dof/.kaggle/_kernel"
    shutil.rmtree(stage_k, ignore_errors=True)
    stage_k.mkdir(parents=True)
    # Stamp the kernel with the fingerprint of the bundle this push uploads, so the job
    # can refuse a stale dataset mount rather than returning the previous run's answer
    # under the new run's name. A silent stale read is the failure that MIMICS the
    # interesting result — see EXPECTED_FINGERPRINT in fp.py.
    source = (KAGGLE / "fp.py").read_text()
    stamped = source.replace("EXPECTED_FINGERPRINT = None",
                             f'EXPECTED_FINGERPRINT = "{meta["fingerprint"]}"', 1)
    if stamped == source:
        raise SystemExit("could not stamp EXPECTED_FINGERPRINT into fp.py — the "
                         "placeholder line has moved; fix it rather than pushing blind")
    # The mesh control must run the functions that built the chair's mesh, verbatim.
    ourway = ourway_source()
    before = stamped
    stamped = stamped.replace("OURWAY_SOURCE = None", f"OURWAY_SOURCE = {ourway!r}", 1)
    if stamped == before:
        raise SystemExit("could not stamp OURWAY_SOURCE into fp.py — placeholder moved")
    targets = target_bundles()
    expected_t = {n: json.load(open(d / "bundle.json"))["fingerprint"]
                  for n, d in sorted(targets.items())}
    before = stamped
    stamped = stamped.replace("EXPECTED_TARGETS = {}", f"EXPECTED_TARGETS = {expected_t!r}", 1)
    if targets and stamped == before:
        raise SystemExit("could not stamp EXPECTED_TARGETS into fp.py — placeholder moved")
    (stage_k / "fp.py").write_text(stamped)
    km = json.loads((KAGGLE / "kernel-metadata.json").read_text())
    km["id"] = f"{username}/{SLUG}"
    km["dataset_sources"] = [s.replace("trishli/", f"{username}/")
                             for s in km.get("dataset_sources", [])]
    (stage_k / "kernel-metadata.json").write_text(json.dumps(km, indent=1))
    print(f"[push]  kernel -> {username}/{SLUG}, "
          f"sources {km['dataset_sources']}")
    if a.dry_run:
        print("[dry-run]  staged and verified; stopping before authentication. "
              "Re-run without --dry-run in a shell that has the token exported.")
        return 0
    print(api.kernels_push(str(stage_k)))

    deadline = time.time() + a.poll_minutes * 60
    print(f"[wait]  polling (GPU kernels queue; giving it {a.poll_minutes:.0f} min)")
    state = "unknown"
    while time.time() < deadline:
        time.sleep(20)
        status = api.kernels_status(f"{username}/{SLUG}")
        # The client returns an enum whose str() is "KernelWorkerStatus.ERROR", so a
        # bare .lower() comparison never matches and the loop polls to the deadline.
        # `grootN1_Robotics/tools/kaggle_baseline.py` already got this right.
        state = str(getattr(status, "status", status)).split(".")[-1].lower()
        print(f"   {state}", flush=True)
        if state in {"complete", "error", "cancelrequested", "cancelled"}:
            break

    out = ROOT / "foundationpose_6dof/.kaggle/_output"
    out.mkdir(exist_ok=True)
    api.kernels_output(f"{username}/{SLUG}", str(out))
    print(f"[done]  output in {out}")

    # The map-prior result travels under the same fingerprint discipline as the main one.
    prior = out / "pose_result_prior.json"
    if prior.exists():
        got_p = json.load(open(prior)).get("bundle_fingerprint")
        if got_p == meta["fingerprint"]:
            shutil.copy2(prior, BUNDLE / "pose_result_prior.json")
            print(f"[done]  map-prior result copied (fingerprint {got_p})")
        else:
            print(f"[done]  REFUSED to copy the map-prior result: {got_p} != {meta['fingerprint']}")

    for name, d in target_bundles().items():
        r = out / f"pose_result_{name}.json"
        if not r.exists():
            print(f"[done]  target {name}: no result in the output")
            continue
        got_t = json.load(open(r)).get("bundle_fingerprint")
        want_t = json.load(open(d / "bundle.json"))["fingerprint"]
        if got_t == want_t:
            shutil.copy2(r, d / "pose_result.json")
            print(f"[done]  target {name}: result copied (fingerprint {got_t})")
        else:
            print(f"[done]  target {name}: REFUSED to copy, {got_t} != {want_t}")

    for extra in ("iteration_trace.json", "iteration_trace_mustard0.json"):
        if (out / extra).exists():
            print(f"[done]  {extra} retrieved")

    result = out / "pose_result.json"
    if result.exists():
        got = json.load(open(result)).get("bundle_fingerprint")
        if got == meta["fingerprint"]:
            shutil.copy2(result, BUNDLE / "pose_result.json")
            print(f"[done]  fingerprint {got} matches; copied to the bundle. "
                  f"Now: conda run -n foundationpose_vl python pipeline/tools/e2e_pose.py")
        else:
            print(f"[done]  REFUSED to copy: result says {got}, bundle says "
                  f"{meta['fingerprint']}. The kernel read a different input than the "
                  f"one on disk — most likely the dataset version did not land.")
            return 1
    else:
        print("[done]  no pose_result.json in the output; read the log above")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
