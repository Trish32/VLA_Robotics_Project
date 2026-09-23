# ORB-SLAM3 — camera localization

Upstream [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) pinned at `4452a3c`, built
on Apple Silicon with **zero patches to upstream source** — portability is handled by
shim headers on the include path, so upstream can be re-pulled and diffed.

**This one clears the Fidelity Rule.** ATE **1.03 cm** on TUM `fr1/xyz` against a
published figure of ~1.0 cm, 798/798 frames tracked, 41.4 FPS on CPU.

**→ [RESULTS.md](RESULTS.md)** — ATE, the dynamic-rejection ablation, what each half costs
**→ [Plan.md](Plan.md)** — what is not measured

## Two things it does beyond building

**Dynamic-feature rejection.** A YOLOv8n + ByteTrack pass marks people, and a
semantic-prior-guided epipolar test plus a 3-D rigid residual confirms which are actually
moving before their features are masked. On `fr3/walking_xyz` that takes ATE from
**80.92 cm to 18.70 cm (−76.9%)** — at a measured cost of 38.1 → 18.5 FPS, which is why
the filtered path is **not** described as real-time.

**A Python binding.** `bindings/orbslam3_py.cpp` exposes `Tracker` via pybind11, so the
rest of the stack drives SLAM from Python without a ROS dependency.

## Layout

```
dynamic_filter.py       epipolar + 3-D rigid motion verification, semantic-prior guided
bindings/               pybind11 Tracker; returns Twc, clones cv::Mat (use-after-free fix)
compat_include/         shim headers -- how upstream stays unpatched
tools/
  build_macos.sh        encodes 8 build findings; run this first
  run_tum_rgbd.py       SLAM runner, --reject-dynamic
  eval_ate.py           ATE with SE(3) (default) or Sim(3) alignment
  detect_dynamic.py     cached YOLO+ByteTrack pass
```

Pangolin is a build-time dependency, cloned and compiled locally by `build_macos.sh`. It
is upstream code and is **not vendored here**.

`tests/` — 18 tests. `bug_log.txt` — 8 root-caused bugs, including a SIGBUS at frame ~700
that took two wrong diagnoses before the real cause (OpenCV's thread pool shared with
ORB-SLAM3's background threads; `cv2.setNumThreads(0)`).
