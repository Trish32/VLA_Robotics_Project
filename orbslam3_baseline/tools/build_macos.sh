#!/usr/bin/env bash
# Build ORB-SLAM3 on Apple Silicon, headless, against conda-forge dependencies.
#
# This is a CPU SLAM system with no CUDA anywhere, which makes it the only tracker in
# this repo that can actually RUN on the Mac. That is the whole point of having it:
# DROID-SLAM's frontend is cloud-verified-pending and always will be locally, so
# ORB-SLAM3 is the baseline we can execute and measure here.
#
# Four things were learned the hard way and are encoded below.
#
# 1. DO NOT use `brew install opencv`. On macOS it pulls OpenVINO and builds it from
#    source -- hours of compilation for DNN acceleration ORB-SLAM3 never calls. The
#    conda-forge opencv is a binary package with none of that.
#
# 2. DO NOT use `brew install eigen`. Homebrew now ships Eigen 5.x; ORB-SLAM3 is written
#    against the Eigen 3 API. `find_package(Eigen3 3.1.0)` is happy to accept 5.0 and
#    then the compile fails deep inside template expansion. Pinned to 3.4 here.
#
# 3. CMake 4 hard-rejects `cmake_minimum_required(VERSION 2.6)`, which ORB_SLAM3,
#    Thirdparty/g2o and Thirdparty/DBoW2 all declare. CMAKE_POLICY_VERSION_MINIMUM=3.5
#    is CMake's own escape hatch and means upstream needs no patch for it.
#
# 4. `-march=native` in the RELEASE flags is FINE on Apple clang 15 / arm64. It is
#    widely reported as a blocker on older toolchains; it was tested here and accepted,
#    so it is deliberately left alone.
#
# Usage:  bash orbslam3_baseline/tools/build_macos.sh [--jobs N]
set -euo pipefail

JOBS="${JOBS:-$(sysctl -n hw.ncpu)}"
[[ "${1:-}" == "--jobs" ]] && JOBS="$2"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="$(conda info --base)/envs/orbslam3_build"
UPSTREAM="${HERE}/upstream"
PANGOLIN_SRC="${HERE}/pangolin-src"
PANGOLIN_PREFIX="${HERE}/pangolin-install"

[[ -d "${PREFIX}" ]] || {
  echo "missing conda env 'orbslam3_build'. Create it with:"
  echo "  conda create -y -n orbslam3_build -c conda-forge 'opencv>=4.8' 'eigen=3.4' \\"
  echo "      glew boost-cpp cmake pybind11 ninja"
  exit 1
}

export CMAKE_POLICY_VERSION_MINIMUM=3.5          # see note 3
export CMAKE_PREFIX_PATH="${PREFIX}:${PANGOLIN_PREFIX}:${CMAKE_PREFIX_PATH:-}"
export PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
CMAKE="${PREFIX}/bin/cmake"

# Thirdparty/DBoW2 includes <boost/serialization/...> in its headers but its
# CMakeLists.txt never calls find_package(Boost) — ORB-SLAM3's fork added the include
# and not the dependency. That works on Linux, where boost lives in /usr/include, and
# fails here. CMAKE_PREFIX_PATH does not help: it feeds find_package, not bare includes.
# So the conda prefix is put on the compiler's search path directly, as -isystem to keep
# boost's own warnings out of our build output.
common_args=(
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}"
  -DEIGEN3_INCLUDE_DIR="${PREFIX}/include/eigen3"
  -DOpenCV_DIR="${PREFIX}/lib/cmake/opencv4"
  # compat_include supplies <stdint-gcc.h>, a GCC-internal header that clang does not
  # ship and that DBoW2/FORB.cpp and src/ORBmatcher.cc both include. Providing it here
  # keeps upstream/ byte-identical to the pinned commit rather than carrying a patch for
  # a one-line portability bug.
  -DCMAKE_CXX_FLAGS="-isystem ${HERE}/compat_include -isystem ${PREFIX}/include"
  -DCMAKE_SHARED_LINKER_FLAGS="-L${PREFIX}/lib -lboost_serialization"
  -DCMAKE_EXE_LINKER_FLAGS="-L${PREFIX}/lib -lboost_serialization"
  -DCMAKE_INSTALL_RPATH="${PREFIX}/lib"
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
)

step() { printf '\n\033[1m[build] %s\033[0m\n' "$1"; }

# ------------------------------------------------------------------- Pangolin
# ORB-SLAM3's CMakeLists says `find_package(Pangolin REQUIRED)` even though the viewer
# is a RUNTIME flag (`System(..., bUseViewer=false)`). So it is needed to link, not to
# run, and we install it into a local prefix rather than polluting the conda env.
if [[ ! -f "${PANGOLIN_PREFIX}/lib/cmake/Pangolin/PangolinConfig.cmake" ]]; then
  step "Pangolin -> ${PANGOLIN_PREFIX}"
  [[ -d "${PANGOLIN_SRC}" ]] || git clone --depth 1 --branch v0.9.1 \
      https://github.com/stevenlovegrove/Pangolin.git "${PANGOLIN_SRC}"
  # ORB-SLAM3's Viewer only uses pango_display / pango_windowing / pango_opengl. Every
  # video driver below is dead weight, and ffmpeg is worse than that: Pangolin 0.9.1's
  # driver calls avcodec_close(), which ffmpeg 7 removed, so conda-forge's ffmpeg being
  # on the include path is enough to break the whole build. Found the hard way --
  # pango_video failed at 81% with "cannot initialize a parameter of type
  # 'AVIOContext *' with an lvalue of type 'AVCodecContext *'".
  "${CMAKE}" -S "${PANGOLIN_SRC}" -B "${PANGOLIN_SRC}/build" "${common_args[@]}" \
      -DCMAKE_INSTALL_PREFIX="${PANGOLIN_PREFIX}" \
      -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF -DBUILD_TOOLS=OFF \
      -DBUILD_PANGOLIN_PYTHON=OFF \
      -DBUILD_PANGOLIN_FFMPEG=OFF -DBUILD_PANGOLIN_LIBOPENEXR=OFF \
      -DBUILD_PANGOLIN_REALSENSE=OFF -DBUILD_PANGOLIN_REALSENSE2=OFF \
      -DBUILD_PANGOLIN_OPENNI=OFF -DBUILD_PANGOLIN_OPENNI2=OFF \
      -DBUILD_PANGOLIN_LIBUVC=OFF -DBUILD_PANGOLIN_LIBDC1394=OFF \
      -DBUILD_PANGOLIN_DEPTHSENSE=OFF -DBUILD_PANGOLIN_TELICAM=OFF \
      -DBUILD_PANGOLIN_PLEORA=OFF -DBUILD_PANGOLIN_LIBRAW=OFF
  "${CMAKE}" --build "${PANGOLIN_SRC}/build" -j "${JOBS}" --target install
else
  step "Pangolin already installed"
fi

# ---------------------------------------------------------------- Thirdparty
for dep in DBoW2 g2o; do
  libdir="${UPSTREAM}/Thirdparty/${dep}/lib"
  if [[ ! -f "${libdir}/lib${dep}.dylib" ]]; then
    step "Thirdparty/${dep}"
    "${CMAKE}" -S "${UPSTREAM}/Thirdparty/${dep}" -B "${UPSTREAM}/Thirdparty/${dep}/build" \
        "${common_args[@]}"
    "${CMAKE}" --build "${UPSTREAM}/Thirdparty/${dep}/build" -j "${JOBS}"
  else
    step "Thirdparty/${dep} already built"
  fi
  # ORB_SLAM3's CMakeLists names these dependencies with a hardcoded `.so` suffix, but
  # CMake produces `.dylib` on macOS, so the link step fails with "No rule to make
  # target ... libDBoW2.so". A symlink satisfies it without patching upstream — the
  # extension is a naming convention here, not a format difference.
  ln -sf "lib${dep}.dylib" "${libdir}/lib${dep}.so"
done

# ------------------------------------------------------------------ Vocabulary
# The ORB vocabulary ships compressed; System() reads the .txt and fails unhelpfully if
# it is still an archive.
if [[ ! -f "${UPSTREAM}/Vocabulary/ORBvoc.txt" ]]; then
  step "extracting ORBvoc.txt"
  tar -xzf "${UPSTREAM}/Vocabulary/ORBvoc.txt.tar.gz" -C "${UPSTREAM}/Vocabulary/"
fi

# ------------------------------------------------------------------ ORB_SLAM3
step "ORB_SLAM3"
"${CMAKE}" -S "${UPSTREAM}" -B "${UPSTREAM}/build" "${common_args[@]}"
"${CMAKE}" --build "${UPSTREAM}/build" -j "${JOBS}"

step "done"
ls -la "${UPSTREAM}/lib/" 2>/dev/null || true
echo
echo "Next: the pybind11 shim exposing System::TrackRGBD over the same ZMQ 'track'"
echo "contract droidSLAM_monocular/slam_server.py speaks, so ros2_bridge's slam_node"
echo "drives either backend from a launch-file switch."
