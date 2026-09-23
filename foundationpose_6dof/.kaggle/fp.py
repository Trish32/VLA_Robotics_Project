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

KERNEL_VERSION = "v12-mustard-control"
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
        "sequence": meta["sequence"],
        "poses_cam_obj": [T.tolist() for T in track],
        "world_spread_cm": spread * 100,
        "translation_residual_cm": float(np.linalg.norm(pose[:3, 3] - origin) * 100),
        "rotation_residual_deg": rot_err_deg(pose[:3, :3], R_ref),
    }, open("/kaggle/working/pose_result.json", "w"), indent=1)
    print("  -> /kaggle/working/pose_result.json", flush=True)
    print("RESULT: REGISTER_OK")
except Exception:
    traceback.print_exc()
    print("RESULT: REGISTER_FAILED")


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

    D = f"{FP}/demo_data"
    # --remaining-ok: the folder holds hundreds of frames and gdown caps at 50 per
    # folder. We need frame 0 and a short track, so the cap is not a problem.
    sh("fetch mustard0", f"cd {FP} && pip -q install gdown && "
       f"gdown --folder --remaining-ok -O {D} "
       f"https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP "
       f"2>&1 | tail -5; find {D} -maxdepth 3 -type d | head -20")

    root = f"{D}/mustard0"
    K_c = np.loadtxt(f"{root}/cam_K.txt").reshape(3, 3)
    mesh_c = trimesh.load(sorted(Path(f"{root}/mesh").glob("*.obj"))[0], process=False)
    rgb_f = sorted(Path(f"{root}/rgb").glob("*.png"))[0]
    dep_f = sorted(Path(f"{root}/depth").glob("*.png"))[0]
    msk_f = sorted(Path(f"{root}/masks").glob("*.png"))[0]

    rgb_c = cv2.cvtColor(cv2.imread(str(rgb_f)), cv2.COLOR_BGR2RGB)
    dep_c = cv2.imread(str(dep_f), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
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
               "behind_surface_cm": behind, "mask_px": int(msk_c.sum())},
              open("/kaggle/working/control_mustard0.json", "w"), indent=1)
    print("RESULT: CONTROL_OK")
except Exception:
    traceback.print_exc()
    print("RESULT: CONTROL_FAILED")
