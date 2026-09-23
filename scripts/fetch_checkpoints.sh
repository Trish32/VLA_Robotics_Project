#!/usr/bin/env bash
# Fetch the weights the two new fidelity gates block on.
#
# Nothing here is tracked: .gitignore excludes checkpoints/ and *.ckpt/*.pth at any
# depth. Re-running is cheap -- every step skips a file that is already present with a
# plausible size, so a partial download can be resumed by just running it again.
#
#   OpenMask3D      Mask3D class-agnostic mask module (two variants) + SAM ViT-H
#   FoundationPose  refiner 2023-10-28-18-33-37 + scorer 2024-01-11-20-02-45
#
# SAM comes from Meta's CDN rather than the Drive mirror OpenMask3D links: it is a
# direct URL with no quota, and it is the same file.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OM3D="${ROOT}/openmask3d_semantic/checkpoints"
FP="${ROOT}/foundationpose_6dof/checkpoints"
mkdir -p "${OM3D}" "${FP}"

GDOWN=(conda run --no-capture-output -n groot_vl python -m gdown)

have() {  # have <path> <min_megabytes>
  [[ -f "$1" ]] && [[ $(( $(stat -f%z "$1" 2>/dev/null || echo 0) / 1048576 )) -ge "$2" ]]
}

report() { printf '%-52s %s\n' "$1" "$2"; }

# ----------------------------------------------------------------- OpenMask3D
if have "${OM3D}/sam_vit_h_4b8939.pth" 2000; then
  report "sam_vit_h_4b8939.pth" "already present"
else
  echo "[fetch] SAM ViT-H (~2.4G) from Meta CDN"
  curl -fL --retry 3 -C - -o "${OM3D}/sam_vit_h_4b8939.pth" \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
fi

# Two mask modules, and they are not interchangeable:
#   scannet200_model.ckpt  -- trained on the ScanNet200 train split. This is the one the
#                             published metric was measured with, so it is what the
#                             Fidelity Rule needs.
#   arbitrary_scene.ckpt   -- the checkpoint upstream recommends for scenes outside
#                             ScanNet, which is what our live RGB-D captures are.
# Using the arbitrary-scene model to "reproduce" the ScanNet200 number would be a
# quietly different experiment, so both are pulled and kept distinct.
if have "${OM3D}/scannet200_model.ckpt" 100; then
  report "scannet200_model.ckpt" "already present"
else
  echo "[fetch] Mask3D scannet200_model.ckpt"
  "${GDOWN[@]}" 1emtZ9xCiCuXtkcGO3iIzIRzcmZAFfI_B -O "${OM3D}/scannet200_model.ckpt"
fi

if have "${OM3D}/arbitrary_scene.ckpt" 100; then
  report "arbitrary_scene.ckpt" "already present"
else
  echo "[fetch] Mask3D arbitrary-scene checkpoint"
  "${GDOWN[@]}" 1rD2Uvbsi89X4lSkont_jUTT7X9iaox9y -O "${OM3D}/arbitrary_scene.ckpt"
fi

# ------------------------------------------------------------- FoundationPose
# readme.md: "For the refiner, you will need 2023-10-28-18-33-37. For scorer, you will
# need 2024-01-11-20-02-45." Both live in one Drive folder; pull the folder and keep
# upstream's directory names, because estimater.py resolves them by name.
if have "${FP}/2023-10-28-18-33-37/model_best.pth" 50 \
   && have "${FP}/2024-01-11-20-02-45/model_best.pth" 50; then
  report "FoundationPose refiner + scorer" "already present"
else
  echo "[fetch] FoundationPose weights folder"
  # No --remaining-ok: gdown 6.1.0 does not have it. The folder holds few enough files
  # that the >50 cap it guards against does not apply.
  "${GDOWN[@]}" --folder 1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i -O "${FP}"
fi

# ------------------------------------------------------------------- summary
echo
echo "=== on disk ==="
find "${OM3D}" "${FP}" -type f \( -name '*.pth' -o -name '*.ckpt' -o -name '*.yml' \) \
  -exec du -h {} + 2>/dev/null | sort -k2
