# FoundationPose on a T4, round 5: import chain + real checkpoint construction.
#
# Where the previous rounds got to, so the remaining risk is visible:
#   r2  nvdiffrast builds and rasterises              RESULT: NVDIFFRAST_OK
#   r3  pinned 'numpy<2' -> broke trimesh's ABI       (self-inflicted; pin removed)
#   r4  trimesh ok, pytorch3d CPU-only ok             failed on a MISSING open3d
#
# r4's failure was a plain absent dependency, so rather than discover the rest one
# 40-minute round at a time, the full set of unguarded module-scope third-party imports
# was extracted from upstream with ast (Utils, estimater, datareader, the two predictors
# and their networks). That is the list below. Anything NOT in it is guarded upstream:
# mycpp, bundlesdf.mycuda, kornia and warp sit behind try/except, and kaolin is imported
# inside the octree functions, which are the model-free NeRF path only.
#
# This round also constructs the predictors against the real weights. That is the part
# that matters: importing proves the environment, constructing proves the CHECKPOINTS
# load, which is the Fidelity Rule's first half and cannot be checked on the Mac.
import os, subprocess, sys, torch, traceback

KERNEL_VERSION = "v19-iteration-trace"

# Stamped by tools/push_kaggle_pose.py at push time with the fingerprint of the bundle
# it uploaded. The job then asserts that the dataset Kaggle actually mounted is that one.
#
# This exists because of a failure mode that MIMICS the interesting result. If the job
# silently reads a stale dataset version -- a cached mount, an upload that did not land,
# a path still pointing at the previous bundle -- then register() returns the OLD answer,
# and an unchanged 175.63 deg reads as "the rebuilt input changed nothing, so the mesh
# geometry is the limit". That conclusion would be unearned and the run would look like
# evidence for it.
#
# The tell is PRECISION. Genuinely different input through genuinely bad geometry lands
# NEAR the old number, essentially never ON it to the same decimals. So a bit-identical
# repeat means "same input" long before it means "same geometry". Rather than rely on
# noticing that after the fact, the job proves which bundle it read, in the output.
EXPECTED_FINGERPRINT = None          # replaced at push time; None = pushed by hand

# Stamped by tools/push_kaggle_pose.py with the VERBATIM source of the functions that
# built the chair's mesh -- `observations.reject_outliers`, `build_pose_bundle.
# poisson_mesh` -- and `foundationpose_6dof/mesh_control.py`. The mesh control below asks
# whether OUR mesh pipeline is the limit, so it must run our code, not a re-typing of it.
OURWAY_SOURCE = None

# v17 answered the mesh control; v18 does not repeat it. Flip to re-run.
RUN_MESH_CONTROL = False


class _Skipped(Exception):
    pass

print(f"=== {KERNEL_VERSION} ===", flush=True)

def sh(label, cmd, tail=2500):
    print(f"\n{'='*70}\n[{label}]\n{'='*70}", flush=True)
    r = subprocess.run(["bash","-lc",cmd], capture_output=True, text=True)
    out = ((r.stdout or "")+(r.stderr or "")).strip()
    print(out[-tail:] if out else "(no output)")
    print(f"--- rc={r.returncode}", flush=True)
    return r.returncode

print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU", flush=True)
if not torch.cuda.is_available():
    print("RESULT: NO_GPU"); raise SystemExit(0)

COMMIT = "a1b694b"
FP = "/kaggle/working/FoundationPose"

sh("system deps", "apt-get -qq update >/dev/null 2>&1; "
   "apt-get -qq install -y libgl1 libglib2.0-0 libgomp1 libusb-1.0-0 >/dev/null 2>&1; echo ok")

# NumPy is the recurring fault line, three rounds running, so it is now GUARDED rather
# than reasoned about. r3 pinned numpy<2 and broke trimesh's C ABI. r5 installed a set
# that MOVED numpy again -- a resolver picking a different version to satisfy open3d or
# pandas -- and the symptom changed to "'numpy.ufunc' object has no attribute
# '__qualname__'", which is the pure-Python half of the same mismatch.
#
# So: record the image's numpy, install, and if anything moved it, put it back. The
# image wins because every preinstalled C extension was compiled against it. Packages
# Kaggle already ships (pandas, matplotlib, pillow, psutil, tqdm, scipy) are not
# reinstalled at all -- that is what dragged the resolver in last time.
sh("baseline numpy", "python -c \"import numpy; print(numpy.__version__)\" > /tmp/np.txt; "
   "cat /tmp/np.txt")
sh("python deps", "pip install -q setuptools wheel ninja "
   "trimesh kornia omegaconf ruamel.yaml warp-lang imageio joblib transformations "
   "2>&1 | tail -6; echo installed")
# open3d is the most aggressive numpy pinner in the set, so it goes in alone and last.
sh("open3d", "pip install -q open3d 2>&1 | tail -4; echo installed")
sh("numpy guard", "WANT=$(cat /tmp/np.txt); "
   "HAVE=$(python -c 'import numpy; print(numpy.__version__)'); "
   "echo \"image numpy=$WANT now=$HAVE\"; "
   "if [ \"$WANT\" != \"$HAVE\" ]; then "
   "  echo 'RESTORING image numpy'; pip install -q --force-reinstall \"numpy==$WANT\"; "
   "fi; python -c 'import numpy; print(\"final numpy\", numpy.__version__)'")
sh("versions", "python -c \"import importlib;"
   "[print(m, getattr(importlib.import_module(m),'__version__','?')) "
   "for m in ('numpy','trimesh','open3d','scipy','pandas','matplotlib')]\" "
   "2>&1 | tail -8")

# Report EVERY missing import at once rather than one per round.
print(f"\n{'='*70}\n[dependency check]\n{'='*70}", flush=True)
missing = []
for mod in ("PIL","cv2","imageio","joblib","kornia","matplotlib","numpy","omegaconf",
            "open3d","pandas","psutil","scipy","torch","torchvision","tqdm",
            "transformations","trimesh","yaml","ruamel.yaml"):
    try:
        __import__(mod)
    except Exception as e:
        missing.append(f"{mod} ({type(e).__name__})")
print("missing:", missing or "none", flush=True)

sh("nvdiffrast", "pip install -q --no-build-isolation "
   "git+https://github.com/NVlabs/nvdiffrast.git 2>&1 | tail -4; echo ok")
sh("pytorch3d", "FORCE_CUDA=0 CUDA_HOME= pip install -q --no-build-isolation "
   "'git+https://github.com/facebookresearch/pytorch3d.git@stable' 2>&1 | tail -6; echo ok")

# mycpp is NOT optional, despite sitting behind a try/except in Utils.py. v9 died 100
# lines past that guard with "module 'mycpp' has no attribute 'cluster_poses'":
# `mycpp` is a SOURCE DIRECTORY in the repo, so Python 3's implicit namespace packages
# resolve `import mycpp` to an empty module object instead of raising. The guard sees
# success, mycpp is not None, and the failure surfaces later as a missing attribute.
# So it has to be compiled, and the compiled module has to be the one that gets imported.
sh("mycpp deps", "apt-get -qq install -y libboost-all-dev libeigen3-dev libomp-dev "
   ">/dev/null 2>&1; pip install -q 'pybind11[global]' 2>&1 | tail -2; echo ok")

sh("clone", f"cd /kaggle/working && rm -rf FoundationPose && "
   f"git clone -q https://github.com/NVlabs/FoundationPose.git && cd FoundationPose && "
   f"git checkout -q {COMMIT} && git rev-parse --short HEAD")

# Upstream resolves weights as learning/training/../../weights/<run_name>/model_best.pth.
# The dataset was uploaded with --dir-mode zip, so handle both extracted and zipped.
sh("weights", f"""
set -e
mkdir -p {FP}/weights
SRC=/kaggle/input/foundationpose-weights
ls -la $SRC
for name in 2023-10-28-18-33-37 2024-01-11-20-02-45; do
  if [ -d "$SRC/$name" ]; then cp -r "$SRC/$name" {FP}/weights/;
  elif [ -f "$SRC/$name.zip" ]; then unzip -qo "$SRC/$name.zip" -d {FP}/weights/$name;
  fi
done
find {FP}/weights -maxdepth 2 | sort
""")

sh("mycpp build", f"""
cd {FP}/mycpp && mkdir -p build && cd build && \
  cmake .. -DPYTHON_EXECUTABLE=$(which python3) \
           -DCMAKE_PREFIX_PATH=$(python3 -m pybind11 --cmakedir) >/dev/null 2>&1 && \
  make -j4 2>&1 | tail -5
ls -la {FP}/mycpp/build/*.so 2>/dev/null || echo 'NO .so BUILT'
""")

# Verify the COMPILED module is what imports, not the empty namespace package.
print(f"\n{'='*70}\n[mycpp check]\n{'='*70}", flush=True)
sys.path.insert(0, f"{FP}/mycpp/build")
try:
    import mycpp
    where = getattr(mycpp, "__file__", None)
    has = hasattr(mycpp, "cluster_poses")
    print(f"  mycpp from {where}", flush=True)
    print(f"  cluster_poses present: {has}", flush=True)
    if not has:
        print("  -> this is the empty NAMESPACE PACKAGE, not the built extension")
        print("RESULT: MYCPP_NOT_BUILT"); raise SystemExit(0)
except ImportError:
    print("  mycpp does not import at all")
    print("RESULT: MYCPP_NOT_BUILT"); raise SystemExit(0)

print(f"\n{'='*70}\n[import chain]\n{'='*70}", flush=True)
sys.path.insert(0, FP)
os.chdir(FP)
for name in ("nvdiffrast.torch","pytorch3d.transforms","trimesh","open3d",
             "Utils","datareader","learning.training.predict_score",
             "learning.training.predict_pose_refine","estimater"):
    try:
        __import__(name); print(f"  {name:<42} ok", flush=True)
    except Exception as e:
        print(f"  {name:<42} FAIL  {type(e).__name__}: {str(e)[:200]}", flush=True)
        if name == "estimater":
            traceback.print_exc()
        if name in ("Utils","estimater"):
            print("RESULT: IMPORT_CHAIN_FAILED"); raise SystemExit(0)

# Constructing was round 6's bar. Round 7 goes further: register() on a real mesh and
# mask cut out of OUR OWN map (pipeline/tools/build_pose_bundle.py) -- no CAD model, as
# foundationpose_6dof/bug_log.txt specifies. The mesh comes from the TSDF, the 2-D mask
# is the instance projected into the frame with an occlusion test.
print(f"\n{'='*70}\n[construct predictors]\n{'='*70}", flush=True)
try:
    from estimater import PoseRefinePredictor, ScorePredictor
    scorer = ScorePredictor()
    print("  ScorePredictor constructed", flush=True)
    refiner = PoseRefinePredictor()
    print("  PoseRefinePredictor constructed", flush=True)
    for tag, obj in (("scorer", scorer), ("refiner", refiner)):
        net = getattr(obj, "model", None)
        if net is not None:
            n = sum(p.numel() for p in net.parameters())
            print(f"  {tag}: {n/1e6:.2f} M params, device "
                  f"{next(net.parameters()).device}", flush=True)
    print("  predictors ok", flush=True)
except Exception:
    traceback.print_exc()
    print("RESULT: CONSTRUCT_FAILED"); raise SystemExit(0)

# ------------------------------------------------------------ register + track
# The mesh and mask come from OUR map, not a CAD model: the TSDF supplies geometry and
# the instance mask projected into the frame supplies ob_mask. That only holds because
# the SLAM poses are metric -- monocular scale would make the mesh the wrong SIZE and the
# pose confidently wrong.
print(f"\n{'='*70}\n[register]\n{'='*70}", flush=True)
try:
    import json
    import cv2, numpy as np, trimesh
    import nvdiffrast.torch as dr
    from estimater import FoundationPose

    B = "/kaggle/input/foundationpose-bundle"
    meta = json.load(open(f"{B}/bundle.json"))
    K = np.array(meta["K"], dtype=np.float64)
    mesh = trimesh.load(f"{B}/mesh.obj", process=False)

    # Which bundle did Kaggle actually mount? Printed before anything else so the answer
    # is at the top of the log, and recorded in the result so a reader months later does
    # not have to trust that the right input was used.
    got = meta.get("fingerprint")
    anchor = meta["frames"][0]
    print(f"\n  [input] fingerprint   {got}", flush=True)
    print(f"  [input] expected      {EXPECTED_FINGERPRINT}", flush=True)
    print(f"  [input] anchor frame  {anchor.get('source_index', 'unknown')}, "
          f"{anchor['mask_pixels']} px, "
          f"{anchor.get('visible_fraction', float('nan')):.0%} of the instance", flush=True)
    print(f"  [input] mask px       "
          f"{[f['mask_pixels'] for f in meta['frames']]}", flush=True)
    if EXPECTED_FINGERPRINT and got != EXPECTED_FINGERPRINT:
        raise SystemExit(
            f"\n  STALE INPUT: this job was pushed for bundle {EXPECTED_FINGERPRINT} "
            f"but Kaggle mounted {got}.\n"
            f"  Refusing to run -- a pose computed on the wrong bundle is worse than no "
            f"pose, because it is indistinguishable from a real result.\n"
            f"  Most likely the dataset version did not land before the kernel started.")

    print(f"  target {meta['target']} ({meta['label']})", flush=True)
    print(f"  mesh {len(mesh.vertices)} verts / {len(mesh.faces)} faces, "
          f"extents {np.round(mesh.extents, 3).tolist()} m", flush=True)

    est = FoundationPose(
        model_pts=mesh.vertices.astype(np.float32),
        model_normals=mesh.vertex_normals.astype(np.float32),
        mesh=mesh, scorer=scorer, refiner=refiner,
        glctx=dr.RasterizeCudaContext(), debug=0,
    )
    print("  FoundationPose constructed", flush=True)

    def hypothesis_agreement(est, tag, k=16):
        """Do the top hypotheses AGREE on an orientation?

        r13 measured score spread and read a flat top as "the orientation is
        unidentifiable". The mustard0 control refuted that: upstream's clean data gives
        an EXACT tie at the top (margin 0.0000, 91/252 within 1%) and still returns a
        correct pose. Scores are computed after refinement, so hypotheses that converge
        onto the same pose legitimately tie — a flat top means AGREEMENT, not ambiguity.
        Score spread simply does not separate the two cases.

        What does separate them is whether the refined POSES cluster. `est.poses` holds
        all 252 sorted by score; if the top-k are the same orientation reached from
        different starts, their pairwise geodesic angles are small and the estimate is
        well-determined. If they are scattered, the scorer is picking between genuinely
        different orientations it cannot tell apart.
        """
        s = np.asarray(est.scores.float().cpu())
        P = np.asarray(est.poses.float().cpu()).reshape(-1, 4, 4)[:k, :3, :3]

        def ang(A, B):
            return np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1)))

        vs_top = [float(ang(P[i], P[0])) for i in range(1, len(P))]
        pair = [float(ang(P[i], P[j]))
                for i in range(len(P)) for j in range(i + 1, len(P))]
        near = int((s > s.max() - 0.01 * abs(s.max())).sum())
        print(f"\n  [{tag}] {len(s)} hypotheses; agreement among top-{len(P)}", flush=True)
        print(f"    scores: top-1 {s.max():.4f}  margin {np.sort(s)[-1]-np.sort(s)[-2]:.4f}"
              f"  within 1%: {near}/{len(s)}", flush=True)
        print(f"    rotation vs top-1: median {np.median(vs_top):6.2f} deg   "
              f"max {np.max(vs_top):6.2f} deg", flush=True)
        print(f"    pairwise among top-{len(P)}: median {np.median(pair):6.2f} deg",
              flush=True)
        print(f"    within 15 deg of top-1: {int(sum(a < 15 for a in vs_top))+1}/{len(P)}",
              flush=True)
        return {"n": int(len(s)), "top1": float(s.max()), "within_1pct": near,
                "median_vs_top1_deg": float(np.median(vs_top)),
                "max_vs_top1_deg": float(np.max(vs_top)),
                "median_pairwise_deg": float(np.median(pair)),
                "clustered_within_15deg": int(sum(a < 15 for a in vs_top)) + 1,
                "k": int(len(P))}

    def load(i):
        rgb = cv2.cvtColor(cv2.imread(f"{B}/rgb_{i:03d}.png"), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(f"{B}/depth_{i:03d}.png", cv2.IMREAD_ANYDEPTH)
        depth = depth.astype(np.float32) / meta["depth_scale"]
        depth[(depth < 0.1) | (depth > 4.0)] = 0
        mask = cv2.imread(f"{B}/mask_{i:03d}.png", cv2.IMREAD_GRAYSCALE) > 0
        return rgb, depth, mask

    rgb, depth, mask = load(0)
    print(f"  frame 0: mask {int(mask.sum())} px, depth valid {int((depth>0).sum())} px",
          flush=True)
    pose = np.asarray(est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                                   iteration=5)).reshape(4, 4)

    # `register()` returns the pose of the mesh AS SUPPLIED: reset_object centres it, and
    # estimater.py:233 (`poses[0] @ get_tf_to_centered_mesh()`) un-centres the answer.
    # v11 re-centred the mesh to "compensate" for a subtraction that was already undone,
    # which only moved the measurement point and made the residual worse, 72.1 -> 112.8
    # cm. The bundle is back to a point-centroid origin and that is the reference.
    origin = np.array(meta["frames"][0]["mesh_origin_cam"])
    print(f"\n  register    t = {np.round(pose[:3, 3], 4).tolist()} m", flush=True)
    print(f"  map says    t = {np.round(origin, 4).tolist()} m", flush=True)
    print(f"  translation residual = {np.linalg.norm(pose[:3,3]-origin)*100:6.2f} cm",
          flush=True)

    # The rotation, measured directly rather than inferred from an origin probe. The mesh
    # is cut out of the world cloud unrotated, so its frame is the world frame up to
    # translation and the estimate should reproduce R_world_to_cam.
    def rot_err_deg(A, B):
        return float(np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1))))
    R_ref = np.array(meta["frames"][0]["R_world_to_cam"])
    print(f"  rotation    residual = {rot_err_deg(pose[:3,:3], R_ref):6.2f} deg",
          flush=True)
    spread_ours = hypothesis_agreement(est, "ours")
    # Depth under the mask says where the observed surface is; a pose far behind it is
    # wrong regardless of which reference point is used.
    zs = depth[mask & (depth > 0)]
    print(f"  median depth under mask    = {float(np.median(zs)):.3f} m", flush=True)

    track = [pose]
    for i in range(1, len(meta["frames"])):
        r, d, _ = load(i)
        track.append(np.asarray(est.track_one(rgb=r, depth=d, K=K,
                                              iteration=2)).reshape(4, 4))
    world = np.array([(np.array(meta["frames"][i]["cam_to_world"]) @ T)[:3, 3]
                      for i, T in enumerate(track)])
    spread = float(np.linalg.norm(world - world.mean(axis=0), axis=1).max())
    print(f"\n  tracked {len(track)} frames", flush=True)
    for i, w in enumerate(world):
        print(f"    {i}  world t = {np.round(w, 3).tolist()}", flush=True)
    # A static object fused into a world frame must not move. No ground truth needed.
    # It is a consistency check, NOT an accuracy one: a tracker locked onto a wrong pose
    # holds it just as steadily as one locked onto the right pose.
    print(f"\n  world-frame spread = {spread*100:.2f} cm", flush=True)

    # Write the poses out. Until r12 they were only ever printed, so no downstream stage
    # could consume them and the 6-DoF path stopped at the log.
    json.dump({
        "ok": True, "source": f"kaggle:foundationpose-nvdiffrast {KERNEL_VERSION}",
        "target": meta["target"], "label": meta["label"],
        # Which bundle this was computed from. The pose stage refuses a result whose
        # fingerprint does not match the bundle on disk, so a rebuilt input cannot be
        # silently scored against an old answer.
        "bundle_fingerprint": meta.get("fingerprint"),
        # Observable facts about the input, so which bundle ran is recoverable from the
        # result even if the fingerprint is ever absent or doubted.
        "input": {
            "anchor_source_index": anchor.get("source_index"),
            "anchor_mask_pixels": anchor["mask_pixels"],
            "anchor_visible_fraction": anchor.get("visible_fraction"),
            "mask_pixels": [f["mask_pixels"] for f in meta["frames"]],
            "mesh": [len(mesh.vertices), len(mesh.faces)],
        },
        "sequence": meta["sequence"],
        "poses_cam_obj": [T.tolist() for T in track],
        "world_spread_cm": spread * 100,
        "translation_residual_cm": float(np.linalg.norm(pose[:3, 3] - origin) * 100),
        "rotation_residual_deg": rot_err_deg(pose[:3, :3], R_ref),
        "agreement": spread_ours,
    }, open("/kaggle/working/pose_result.json", "w"), indent=1)
    print("  -> /kaggle/working/pose_result.json", flush=True)
    print("RESULT: REGISTER_OK")
except Exception:
    traceback.print_exc()
    print("RESULT: REGISTER_FAILED")


# ---------------------------------------------------------------------- iteration trace
# v18: one refine call from the map pose (iteration=5 -> five internal updates,
# predict_pose_refine.py:182) moved a pose that fits the depth to 3.8 cm by 77 cm. A
# review asked the question v18 could not answer: is the FIRST update already wrong, or do
# later updates diverge? The fix differs — fewer iterations help only in the second case.
#
# So: from the map pose, call the refiner with iteration=1, ten times in sequence, and
# record every iterate. Depth residual and coverage are computed with OUR mesh and OUR
# mask, by the same function the stage-4 gate uses (stamped verbatim, like the mesh
# control). Every iterate is also scored, together with register()'s best, in one batch.
# Then the pre-registered candidate fix: track all 8 frames from the map pose with
# iteration=1 everywhere, written for the gate.
#
# Conventions as in v18: upstream poses are relative to the CENTRED mesh; ours refer to the
# mesh as supplied, so internally a pose is ours @ T(+centre).
print(f"\n{'='*70}\n[iteration trace: refine from the map pose, one update at a time]\n{'='*70}", flush=True)
try:
    from Utils import erode_depth, bilateral_filter_depth, depth2xyzmap
    ow = {}
    exec(OURWAY_SOURCE, ow)
    depth_agreement_ = ow["depth_agreement"]

    tf_c = est.get_tf_to_centered_mesh().data.cpu().numpy()      # T(-centre)
    to_int = np.linalg.inv(tf_c)                                 # T(+centre)
    # OUR geometry, reloaded from the file: upstream's reset_object may re-centre the
    # trimesh it was handed, and the gate measures against the mesh as supplied.
    _sup = trimesh.load(f"{B}/mesh.obj", process=False)
    V_sup, F_sup = np.asarray(_sup.vertices), np.asarray(_sup.faces)

    def pose_map(i):
        f = meta["frames"][i]
        P = np.eye(4)
        P[:3, :3] = np.asarray(f["R_world_to_cam"])
        P[:3, 3] = f["mesh_origin_cam"]
        return P

    def agreement(P_sup, i):
        _, d, m = load(i)
        return depth_agreement_(V_sup, F_sup, P_sup, K, np.where(m, d, 0.0))

    # sanity: our geometry at the map pose must reproduce the local measurement (3.77 cm)
    r0, c0 = agreement(pose_map(0), 0)
    print(f"  map pose, frame 0: |dz| {r0*100:.2f} cm, cover {c0:.0%} "
          f"(local measurement: 3.77 cm) -- geometry convention check", flush=True)

    rgb0, depth0, mask0 = load(0)
    d_p = bilateral_filter_depth(erode_depth(depth0, radius=2, device="cuda"), radius=2, device="cuda")
    xyz0 = depth2xyzmap(d_p, K)

    def refine(P_int, n):
        out, _ = est.refiner.predict(
            mesh=est.mesh, mesh_tensors=est.mesh_tensors, rgb=rgb0, depth=d_p, K=K,
            ob_in_cams=np.asarray(P_int, np.float32)[None], normal_map=None, xyz_map=xyz0,
            glctx=est.glctx, mesh_diameter=est.diameter, iteration=n, get_vis=False)
        return out.data.cpu().numpy().reshape(4, 4).astype(np.float32)

    def rdeg(A, B):
        return float(np.degrees(np.arccos(np.clip((np.trace(A[:3, :3] @ B[:3, :3].T) - 1) / 2, -1, 1))))

    half_d = float(est.diameter) / 2
    start = (pose_map(0) @ to_int).astype(np.float32)
    iters, cur = [start], start
    for k in range(10):
        cur = refine(cur, 1)
        iters.append(cur)
    five_at_once = refine(start, 5)
    eq_t = float(np.linalg.norm(five_at_once[:3, 3] - iters[5][:3, 3]) * 100)
    eq_r = rdeg(five_at_once, iters[5])
    print(f"  equivalence: 1 x (5 iterations) vs 5 x (1 iteration): {eq_t:.2f} cm, {eq_r:.2f} deg", flush=True)

    fp_int = est.poses[0].data.cpu().numpy().reshape(4, 4).astype(np.float32)
    batch = np.stack([fp_int] + iters)
    sc, _ = est.scorer.predict(
        mesh=est.mesh, rgb=rgb0, depth=d_p, K=K, ob_in_cams=batch, normal_map=None,
        mesh_tensors=est.mesh_tensors, glctx=est.glctx, mesh_diameter=est.diameter,
        get_vis=False)
    sc = np.asarray(sc.float().cpu()).ravel()

    trace = []
    print(f"\n  {'k':>3}{'step cm':>9}{'step/(d/2)':>11}{'step deg':>9}{'vs map cm':>10}"
          f"{'|dz| cm':>9}{'cover':>7}{'score':>10}", flush=True)
    for k, P in enumerate(iters):
        step_t = 0.0 if k == 0 else float(np.linalg.norm(P[:3, 3] - iters[k-1][:3, 3]) * 100)
        step_r = 0.0 if k == 0 else rdeg(P, iters[k-1])
        vs_map = float(np.linalg.norm(P[:3, 3] - start[:3, 3]) * 100)
        r, c = agreement(P @ tf_c, 0)
        trace.append({"k": k, "step_cm": step_t, "step_norm": step_t / 100 / half_d,
                      "step_deg": step_r, "vs_map_cm": vs_map, "dz_cm": r * 100,
                      "coverage": c, "score": float(sc[k + 1])})
        print(f"  {k:3d}{step_t:9.2f}{step_t/100/half_d:11.3f}{step_r:9.2f}{vs_map:10.2f}"
              f"{r*100:9.2f}{c:7.0%}{sc[k+1]:10.4f}", flush=True)
    print(f"  register() best, same batch: score {sc[0]:.4f}", flush=True)

    # -- the pre-registered candidate fix: iteration=1 everywhere, 8 frames, for the gate
    first = iters[1]
    est.pose_last = torch.as_tensor(first, device="cuda", dtype=torch.float)
    track1 = [first @ tf_c]
    for i in range(1, len(meta["frames"])):
        r_, d_, _ = load(i)
        track1.append(np.asarray(est.track_one(rgb=r_, depth=d_, K=K, iteration=1)).reshape(4, 4))
    json.dump({
        "ok": True, "source": f"kaggle:foundationpose-nvdiffrast {KERNEL_VERSION}",
        "initialization": "map_prior", "iterations_per_frame": 1,
        "target": meta["target"], "label": meta["label"],
        "bundle_fingerprint": meta.get("fingerprint"), "sequence": meta["sequence"],
        "poses_cam_obj": [np.asarray(T).tolist() for T in track1],
    }, open("/kaggle/working/pose_result_prior.json", "w"), indent=1)
    json.dump({"map_pose_check": {"dz_cm": r0 * 100, "coverage": c0},
               "equivalence": {"cm": eq_t, "deg": eq_r},
               "half_diameter_m": half_d, "register_best_score": float(sc[0]),
               "trace": trace},
              open("/kaggle/working/iteration_trace.json", "w"), indent=1)
    print("RESULT: TRACE_OK", flush=True)
except Exception:
    traceback.print_exc()
    print("RESULT: TRACE_FAILED", flush=True)


# ---------------------------------------------------------------------------- control
# The Fidelity Rule test, and it is deliberately LAST so it cannot take the primary
# result down with it.
#
# Our own bundle gives a 72 cm translation residual and a >= 99 deg rotation error. That
# is either (a) our TSDF mesh and 695-px mask being too poor to fix an orientation, or
# (b) this adaptation being broken. Nothing measured so far separates those, because
# every number came from our own data.
#
# mustard0 is upstream's own demo: a clean CAD mesh, a full mask, a sequence the authors
# publish results on. Running it on the SAME code path answers the question directly.
# A small residual here convicts our bundle; a large one convicts the port.
print(f"\n{'='*70}\n[control: upstream demo_data/mustard0]\n{'='*70}", flush=True)
try:
    import numpy as np, cv2, trimesh, json
    import nvdiffrast.torch as dr
    from pathlib import Path
    from estimater import FoundationPose

    D = "/tmp/demo_data"
    # r12 downloaded this and stopped one step short: the Drive "folder" holds a single
    # 362 MB mustard0.zip, so gdown --folder fetched an archive and the reader then
    # looked for demo_data/mustard0/cam_K.txt inside a directory that did not exist.
    sh("fetch mustard0", f"cd {FP} && pip -q install gdown && "
       f"[ -f {D}/mustard0.zip ] || gdown --folder --remaining-ok -O {D} "
       f"https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP "
       f"2>&1 | tail -3; "
       f"for z in {D}/*.zip; do unzip -qo \"$z\" -d {D}/; done; "
       f"find {D} -maxdepth 2 -type d | head -20")

    root = f"{D}/mustard0"
    if not Path(f"{root}/cam_K.txt").exists():
        # The archive's internal layout is upstream's to choose, not ours to assume.
        hits = list(Path(D).rglob("cam_K.txt"))
        if not hits:
            raise SystemExit(f"no cam_K.txt anywhere under {D} — layout changed")
        root = str(hits[0].parent)
        print(f"  (mustard0 unpacked to {root})", flush=True)
    K_c = np.loadtxt(f"{root}/cam_K.txt").reshape(3, 3)
    mesh_c = trimesh.load(sorted(Path(f"{root}/mesh").glob("*.obj"))[0], process=False)
    rgb_f = sorted(Path(f"{root}/rgb").glob("*.png"))[0]
    # Pair depth and mask to the rgb frame BY NAME, as upstream's YcbineoatReader does.
    # Until v16 this took sorted(depth)[0] -- but mustard0 ships 1,332 depth images for
    # 737 rgb, so "first depth" need not be frame 0's. Whether it was is printed, because
    # every earlier claim about this control rests on it.
    dep_f = Path(str(rgb_f).replace("/rgb/", "/depth/"))
    msk_f = Path(str(rgb_f).replace("/rgb/", "/masks/"))
    old_dep = sorted(Path(f"{root}/depth").glob("*.png"))[0]
    print(f"  depth for frame 0: {dep_f.name}; pre-v16 pairing used {old_dep.name} "
          f"-> {'SAME file' if old_dep.name == dep_f.name else 'DIFFERENT file'}", flush=True)

    rgb_c = cv2.cvtColor(cv2.imread(str(rgb_f)), cv2.COLOR_BGR2RGB)
    dep_raw = cv2.imread(str(dep_f), cv2.IMREAD_ANYDEPTH)
    if dep_raw.shape != rgb_c.shape[:2]:              # upstream resizes; so do we
        dep_raw = cv2.resize(dep_raw, rgb_c.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    dep_c = dep_raw.astype(np.float32) / 1000.0
    dep_c[(dep_c < 0.1) | (dep_c > 4.0)] = 0
    msk_c = cv2.imread(str(msk_f), cv2.IMREAD_GRAYSCALE) > 0
    print(f"  mesh {len(mesh_c.vertices)} verts, extents "
          f"{np.round(mesh_c.extents, 3).tolist()} m", flush=True)
    print(f"  frame {rgb_f.name}: mask {int(msk_c.sum())} px "
          f"(ours was 695 px)", flush=True)

    est_c = FoundationPose(
        model_pts=mesh_c.vertices.astype(np.float32),
        model_normals=mesh_c.vertex_normals.astype(np.float32),
        mesh=mesh_c, scorer=scorer, refiner=refiner,
        glctx=dr.RasterizeCudaContext(), debug=0)
    pose_c = np.asarray(est_c.register(K=K_c, rgb=rgb_c, depth=dep_c, ob_mask=msk_c,
                                       iteration=5)).reshape(4, 4)
    print(f"\n  register t = {np.round(pose_c[:3, 3], 4).tolist()} m", flush=True)

    # THE comparison. Same scorer, same code path, same number of hypotheses — the only
    # difference is the quality of the mesh and mask. Ours: range 1.75, std 0.196, with
    # 48/252 hypotheses within 1% of the top. If mustard0 is comparably flat, the port is
    # broken. If it separates cleanly, our bundle is the problem and the orientation was
    # never observable from a one-sided shell behind a 695 px mask.
    spread_c = hypothesis_agreement(est_c, "mustard0")

    # No annotation needed for the check that matters here: a correct pose puts the
    # object's origin about half its depth behind the surface the depth camera sees.
    z_obs = float(np.median(dep_c[msk_c & (dep_c > 0)]))
    behind = (pose_c[2, 3] - z_obs) * 100
    half_depth = float(mesh_c.extents.max()) / 2 * 100
    print(f"  median depth under mask = {z_obs:.3f} m", flush=True)
    print(f"  origin sits {behind:+.2f} cm behind it "
          f"(half the mesh depth is {half_depth:.1f} cm)", flush=True)

    gt_dir = Path(f"{root}/annotated_poses")
    if gt_dir.is_dir() and (gt := sorted(gt_dir.glob("*.txt"))):
        T_gt = np.loadtxt(gt[0]).reshape(4, 4)
        d = np.linalg.norm(pose_c[:3, 3] - T_gt[:3, 3]) * 100
        r = float(np.degrees(np.arccos(np.clip(
            (np.trace(pose_c[:3, :3] @ T_gt[:3, :3].T) - 1) / 2, -1, 1))))
        print(f"\n  vs annotated pose: {d:.2f} cm, {r:.2f} deg   <- ground truth",
              flush=True)
    else:
        print("\n  no annotated_poses/ in the download — depth check only", flush=True)

    json.dump({"ok": True, "pose": pose_c.tolist(), "median_depth_m": z_obs,
               "behind_surface_cm": float(behind), "mask_px": int(msk_c.sum()),
               "agreement": spread_c},
              open("/kaggle/working/control_mustard0.json", "w"), indent=1)
    print("RESULT: CONTROL_OK")
except Exception:
    traceback.print_exc()
    print("RESULT: CONTROL_FAILED")

# What a HEALTHY refiner does from a correct pose: mustard0, from its validated CAD pose,
# one update at a time. The chair's first step is read against this, in units of the
# mesh's half-diameter, because that is the unit the refiner's translation is scaled by
# (predict_pose_refine.py:229).
print(f"\n{'='*70}\n[iteration trace, healthy baseline: mustard0 from its CAD pose]\n{'='*70}", flush=True)
try:
    from Utils import erode_depth, bilateral_filter_depth, depth2xyzmap
    d_pc = bilateral_filter_depth(erode_depth(dep_c, radius=2, device="cuda"), radius=2, device="cuda")
    xyz_c = depth2xyzmap(d_pc, K_c)
    cur_c = est_c.poses[0].data.cpu().numpy().reshape(4, 4).astype(np.float32)
    half_c = float(est_c.diameter) / 2
    base = []
    for k in range(1, 6):
        nxt, _ = est_c.refiner.predict(
            mesh=est_c.mesh, mesh_tensors=est_c.mesh_tensors, rgb=rgb_c, depth=d_pc, K=K_c,
            ob_in_cams=cur_c[None], normal_map=None, xyz_map=xyz_c, glctx=est_c.glctx,
            mesh_diameter=est_c.diameter, iteration=1, get_vis=False)
        nxt = nxt.data.cpu().numpy().reshape(4, 4).astype(np.float32)
        st = float(np.linalg.norm(nxt[:3, 3] - cur_c[:3, 3]))
        sr = float(np.degrees(np.arccos(np.clip((np.trace(nxt[:3, :3] @ cur_c[:3, :3].T) - 1) / 2, -1, 1))))
        base.append({"k": k, "step_cm": st * 100, "step_norm": st / half_c, "step_deg": sr})
        print(f"  k {k}: step {st*100:.3f} cm = {st/half_c:.4f} half-diameters, {sr:.3f} deg", flush=True)
        cur_c = nxt
    json.dump({"half_diameter_m": half_c, "trace": base},
              open("/kaggle/working/iteration_trace_mustard0.json", "w"), indent=1)
    print("RESULT: BASELINE_TRACE_OK", flush=True)
except Exception:
    traceback.print_exc()
    print("RESULT: BASELINE_TRACE_FAILED", flush=True)


# ---------------------------------------------------------------------------------------
# MESH CONTROL -- is the chair's failure our mesh pipeline, or the chair?
#
# v15 removed every input defect upstream of the mesh and the top-16 hypotheses still
# scattered (1/16 within 15 deg), while mustard0 in the same job converged 16/16 on its
# CAD mesh. Those two differ in mesh AND object. This holds the object fixed and swaps
# only the mesh: mustard0, same frame, same upstream mask, same estimator -- with a mesh
# built the chair's way, from fused depth.
#
# The reference is the CAD mesh's own register() on frame 0, which is the validated pose
# (origin 2.06 cm behind the surface; 16/16 hypotheses agree). mustard0 ships no
# annotated poses, so this is the best reference available, and it is labelled as such.
# CAD tracking over the sequence gives the camera trajectory in the object frame, which
# plays the part SLAM plays for the chair.
# ---------------------------------------------------------------------------------------
print(f"\n{'='*70}\n[mesh control: mustard0 with a mesh built our way]\n{'='*70}", flush=True)
try:
    if not RUN_MESH_CONTROL:
        raise _Skipped()
    if OURWAY_SOURCE is None:
        raise SystemExit("OURWAY_SOURCE was not stamped -- push with tools/push_kaggle_pose.py")
    ow = {}
    exec(OURWAY_SOURCE, ow)
    reject_outliers_, poisson_mesh_ = ow["reject_outliers"], ow["poisson_mesh"]

    rgbs = sorted(Path(f"{root}/rgb").glob("*.png"))
    # Depth is paired by NAME, as upstream does: there are more depth images than rgb
    # (1,332 vs 737), so pairing by sorted index pairs the wrong instants.
    deps = [Path(str(f).replace("/rgb/", "/depth/")) for f in rgbs]
    missing = [d.name for d in deps if not d.exists()]
    msks = [Path(str(f).replace("/rgb/", "/masks/")) for f in rgbs]
    msks = [m for m in msks if m.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} rgb frames have no same-named depth, e.g. {missing[:3]}")
    print(f"  mustard0: {len(rgbs)} rgb, each with its same-named depth; "
          f"{len(msks)} upstream mask(s)", flush=True)

    def frame_c(i):
        rgb = cv2.cvtColor(cv2.imread(str(rgbs[i])), cv2.COLOR_BGR2RGB)
        raw = cv2.imread(str(deps[i]), cv2.IMREAD_ANYDEPTH)
        if raw.shape != rgb.shape[:2]:
            raw = cv2.resize(raw, rgb.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        d = raw.astype(np.float32) / 1000.0
        d[(d < 0.1) | (d > 4.0)] = 0
        return rgb, raw, d

    # -- 1. reference trajectory: CAD tracking over EVERY frame. v16b took every 2nd and
    # the bottle swings from ~4 to ~75 deg quickly, so a 15 or 31 deg arm could otherwise
    # hold almost no frames in between.
    track_ids = list(range(0, min(len(rgbs), 300)))
    P = {0: pose_c}
    depth_m = {}
    for i in track_ids[1:]:
        rgb, _, d = frame_c(i)
        P[i] = np.asarray(est_c.track_one(rgb=rgb, depth=d, K=K_c, iteration=2)).reshape(4, 4)
        depth_m[i] = d
    depth_m[0] = frame_c(0)[2]
    angles = ow["view_angles"]([P[i] for i in track_ids])
    print(f"  tracked {len(track_ids)} frames with the CAD mesh; view angle vs frame 0: "
          f"max {angles.max():.1f} deg, median {np.median(angles):.1f} deg", flush=True)
    hist = np.histogram(angles, bins=[0, 2, 4, 6, 10, 15, 20, 25, 31, 40, 60, 90])
    print("  frames per view-angle bin: " + ", ".join(
        f"{int(a)}-{int(b)}:{n}" for a, b, n in zip(hist[1][:-1], hist[1][1:], hist[0])), flush=True)

    # -- 2. the VOID test, measured directly: does the CAD model at each tracked pose agree
    # with the measured depth? Registered threshold: per-frame median |dz| <= 1.5 cm.
    resid = {i: ow["track_depth_residual"](mesh_c.vertices, mesh_c.faces, P[i], K_c, depth_m[i])
             for i in track_ids}
    r = np.array([resid[i] for i in track_ids])
    print(f"  CAD track vs measured depth, per frame: median {np.median(r)*100:.2f} cm, "
          f"p90 {np.percentile(r, 90)*100:.2f} cm, frames > 1.5 cm: {(r > 0.015).sum()}", flush=True)

    print(f"  fusion masks: {'upstream, per frame' if len(msks) == len(rgbs) else 'frame 0 upstream, rest projected from the CAD track'}", flush=True)

    def cad_mask(i, shape):
        v = mesh_c.vertices @ P[i][:3, :3].T + P[i][:3, 3]
        uv = (v @ K_c.T)
        u = np.round(uv[:, 0] / uv[:, 2]).astype(int); w_ = np.round(uv[:, 1] / uv[:, 2]).astype(int)
        m = np.zeros(shape, np.uint8)
        ok = (u >= 0) & (u < shape[1]) & (w_ >= 0) & (w_ < shape[0])
        hull = cv2.convexHull(np.stack([u[ok], w_[ok]], 1).astype(np.int32))
        cv2.fillConvexPoly(m, hull, 255)
        return m > 0

    longest = float(mesh_c.extents.max())
    voxel = ow["relative"](ow["CHAIR_VOXEL_M"], longest)
    centre = mesh_c.bounds.mean(axis=0)
    print(f"  matched to the chair: voxel {voxel*1000:.2f} mm "
          f"({longest/voxel:.0f} across; chair {1.58/0.02:.0f}), "
          f"point budget {ow['CHAIR_POINTS']}", flush=True)

    def build(ids, tag):
        frames = []
        for i in ids:
            rgb, raw, _ = frame_c(i)
            if len(msks) == len(rgbs):
                m = cv2.imread(str(msks[i]), cv2.IMREAD_GRAYSCALE) > 0
                if m.shape != raw.shape:
                    m = cv2.resize(m.astype(np.uint8), raw.shape[::-1],
                                   interpolation=cv2.INTER_NEAREST) > 0
            else:
                m = msk_c if i == 0 else cad_mask(i, raw.shape)
            frames.append((rgb, raw, m, P[i]))
        pts, cols = ow["fuse_object"](frames, K_c, voxel, centre, 0.6 * longest)
        v, f, vc, c, n = ow["ourway_mesh"](pts, cols, longest, reject_outliers_, poisson_mesh_)
        tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
        tm.visual = trimesh.visual.ColorVisuals(
            tm, vertex_colors=(np.clip(vc, 0, 1) * 255).astype(np.uint8))
        print(f"  [{tag}] {len(ids)} frames fused, {len(pts)} TSDF points -> {n} after thinning "
              f"and rejection, mesh {len(v)} verts / {len(f)} faces, "
              f"extents {np.round(tm.extents, 3).tolist()} m (reported, not a void test)", flush=True)
        return tm, c

    # -- 3. arms. Each takes every tracked frame within theta of frame 0. "wide" is the
    # positive control: it passed in v16b and must pass again or the run is void.
    arms = {f"{th} deg": [i for i, a in zip(track_ids, angles) if a <= th] for th in (4, 15, 31)}
    arms["wide"] = list(track_ids)
    target = {"4 deg": 4, "15 deg": 15, "31 deg": 31, "wide": None}

    rgb0, _, d0 = frame_c(0)
    long_axis = int(np.argmax(mesh_c.extents))
    out = {"reference": "CAD register() on frame 0 -- validated, not ground truth",
           "rule": "committed before this run; see foundationpose_6dof/Plan.md",
           "voxel_m": voxel, "track_residual_m": {"median": float(np.median(r)),
           "p90": float(np.percentile(r, 90))}, "arms": {}}
    for tag, ids in arms.items():
        span = ow["view_span"]([P[i] for i in ids])
        bad = [i for i in ids if resid[i] > 0.015]
        void = []
        if len(ids) < 5:
            void.append(f"{len(ids)} frames < 5")
        if target[tag] is not None and span < 0.8 * target[tag]:
            void.append(f"span {span:.1f} deg < 0.8 x {target[tag]}")
        if len(bad) > 0.2 * len(ids):
            void.append(f"{len(bad)}/{len(ids)} fused frames disagree with depth by > 1.5 cm")
        # Void is decided and printed BEFORE the pose is computed.
        print(f"\n  [{tag}] {len(ids)} frames, view span {span:.1f} deg, "
              f"track-depth disagreements {len(bad)}/{len(ids)} -> "
              f"{'VOID: ' + '; '.join(void) if void else 'valid'}", flush=True)
        rec = {"frames": len(ids), "view_span_deg": span, "void": void,
               "track_disagreements": len(bad)}
        try:
            tm, c = build(ids, tag)
        except Exception as exc:
            print(f"  [{tag}] mesh failed to build: {exc}", flush=True)
            rec["void"] = void + [f"mesh build failed: {exc}"]
            out["arms"][tag] = rec
            continue
        est_o = FoundationPose(
            model_pts=tm.vertices.astype(np.float32),
            model_normals=tm.vertex_normals.astype(np.float32),
            mesh=tm, scorer=scorer, refiner=refiner,
            glctx=dr.RasterizeCudaContext(), debug=0)
        pose_o = np.asarray(est_o.register(K=K_c, rgb=rgb0, depth=d0, ob_mask=msk_c,
                                           iteration=5)).reshape(4, 4)
        ref_t = pose_c[:3, :3] @ c + pose_c[:3, 3]
        rot = ow["rotation_deg"](pose_o, pose_c)
        rot_f = ow["rotation_deg_mod_flip"](pose_o, pose_c, long_axis)
        tr = float(np.linalg.norm(pose_o[:3, 3] - ref_t) * 100)
        agree = hypothesis_agreement(est_o, f"mustard0 / {tag}")
        passed = rot_f <= 30 and agree["clustered_within_15deg"] >= 8
        print(f"  [{tag}] vs the CAD pose: rotation {rot:.2f} deg ({rot_f:.2f} mod flip), "
              f"translation {tr:.2f} cm, {agree['clustered_within_15deg']}/16 cluster -> "
              f"{'PASS' if passed else 'FAIL'}{' (VOID)' if void else ''}", flush=True)
        rec.update(rotation_deg=rot, rotation_deg_mod_flip=rot_f, translation_cm=tr,
                   agreement=agree, passed=bool(passed),
                   mesh=[len(tm.vertices), len(tm.faces)])
        out["arms"][tag] = rec
        del est_o
        torch.cuda.empty_cache()
    json.dump(out, open("/kaggle/working/control_mesh.json", "w"), indent=1)
    print("RESULT: MESH_CONTROL_OK", flush=True)
except _Skipped:
    print("  skipped -- answered in v17 (RUN_MESH_CONTROL = False)", flush=True)
except BaseException:
    traceback.print_exc()
    print("RESULT: MESH_CONTROL_FAILED", flush=True)
