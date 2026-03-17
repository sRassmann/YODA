#!/usr/bin/env bash
set -euo pipefail

# RS example end-to-end pipeline for YODA
# - downloads RS example ZIP from Zenodo
# - builds singularity images (YODA env + FreeSurfer)
# - registers T1/T2 to FLAIR via FreeSurfer tools
# - runs YODA wrapper on the registered images
#
# Requirements:
#   - singularity/apptainer installed and able to build images
#   - NVIDIA GPU + drivers if you want --nv execution (recommended)
#   - a valid FreeSurfer license at $FS_LICENSE (required by FreeSurfer)
#   - run directory containing config.yml and ckpt (see README Inference/Weights)

############################################
# 0) User-controlled paths via env vars
############################################

# Where intermediate/extracted data and outputs go
export PROCESSING_PATH="${PROCESSING_PATH:-/home/${USER}/yoda_processing}"

# Path to YODA run directory (must contain config.yml and ckpt/last.pth unless overridden)
export RUN_DIR="${RUN_DIR:-/home/${USER}/yoda_run}"

# Where singularity .sif images are stored
export SINGULARITY_DIR="${SINGULARITY_DIR:-$HOME/singularity}"

# Optional: checkpoint path (relative to RUN_DIR or absolute)
export CKPT_PATH="${CKPT_PATH:-ckpt/last.pth}"

# Optional: output prediction path
export OUT_PRED="${OUT_PRED:-$PROCESSING_PATH/output/pred.nii.gz}"

# Optional: execution toggles
export USE_NV="${USE_NV:-1}"          # 1 => run singularity with --nv
export THREADS="${THREADS:-16}"

# Optional: set this to 0 if you don't want to build images (assumes they already exist)
export BUILD_IMAGES="${BUILD_IMAGES:-1}"

############################################
# Derived locations
############################################

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RAW_DATA="$PROCESSING_PATH/data/rs_example_raw/subj_0000"
REG_DATA="$PROCESSING_PATH/data/rs_example_registered/subj_0000"

mkdir -p "$PROCESSING_PATH" "$SINGULARITY_DIR" \
  "$(dirname "$RAW_DATA")" "$(dirname "$REG_DATA")" "$(dirname "$OUT_PRED")"

YODA_SIF="$SINGULARITY_DIR/dagobah.sif"
FS_SIF="$SINGULARITY_DIR/freesurfer_74.sif"

# Common sing exec options
SING_NV_ARGS=()
if [[ "$USE_NV" == "1" ]]; then
  SING_NV_ARGS+=("--nv")
fi

# Binds: repo (for wrapper + python code) and processing path (data + outputs)
SING_BINDS=(
  "-B" "$REPO_ROOT:$REPO_ROOT"
  "-B" "$PROCESSING_PATH:$PROCESSING_PATH"
  "-B" "$RUN_DIR:$RUN_DIR":ro
)

run_yoda_py() {
  singularity exec "${SING_NV_ARGS[@]}" "${SING_BINDS[@]}" "$YODA_SIF" python "$@"
}

run_fs() {
  # FreeSurfer tools sometimes want SUBJECTS_DIR; we don't use it here but set it anyway.
  singularity exec "${SING_NV_ARGS[@]}" \
    "${SING_BINDS[@]}"
    -B "$FS_LICENSE:$FS_LICENSE":ro \
    --env SUBJECTS_DIR="$PROCESSING_PATH/freesurfer_subjects" \
    --env FS_LICENSE="${FS_LICENSE:-}" \
    "$FS_SIF" "$@"
}

############################################
# 1) Download RS example data
############################################

ZIP_DIR="$PROCESSING_PATH/downloads"
ZIP_FILE="$ZIP_DIR/sub_rs_mri_raw.zip"
mkdir -p "$ZIP_DIR"

if [[ ! -f "$ZIP_FILE" ]]; then
  echo "[1/5] Downloading RS example ZIP to: $ZIP_FILE"
  wget -O "$ZIP_FILE" "https://zenodo.org/records/11186582/files/sub_rs_mri_raw.zip"
else
  echo "[1/5] ZIP already present: $ZIP_FILE"
fi

############################################
# 2) Unzip (extract only required files)
############################################

echo "[2/5] Extracting T1/T2/FLAIR into: $RAW_DATA"
mkdir -p "$RAW_DATA"

# Only extract if missing
if [[ ! -f "$RAW_DATA/T1_RMS.nii.gz" || ! -f "$RAW_DATA/T2_caipi.nii.gz" || ! -f "$RAW_DATA/FLAIR.nii.gz" ]]; then
  unzip -j -o "$ZIP_FILE" \
    "sub_rs_mri_raw/T1_RMS.nii.gz" \
    "sub_rs_mri_raw/T2_caipi.nii.gz" \
    "sub_rs_mri_raw/FLAIR.nii.gz" \
    -d "$RAW_DATA"
else
  echo "    Inputs already extracted."
fi

############################################
# 3) Build singularity images
############################################

echo "[3/5] Ensuring singularity images are available."

if [[ "$BUILD_IMAGES" == "1" ]]; then
  if [[ ! -f "$YODA_SIF" ]]; then
    echo "    Building YODA env SIF: $YODA_SIF"
    singularity build "$YODA_SIF" docker://srassmann/dif:latest
  else
    echo "    YODA env SIF already exists: $YODA_SIF"
  fi

  if [[ ! -f "$FS_SIF" ]]; then
    echo "    Building FreeSurfer SIF: $FS_SIF"
    singularity build "$FS_SIF" docker://freesurfer/freesurfer:latest
  else
    echo "    FreeSurfer SIF already exists: $FS_SIF"
  fi
else
  echo "    BUILD_IMAGES=0, skipping image builds."
fi

############################################
# 4) Register using FreeSurfer singularity
############################################

echo "[4/5] Registering T1/T2 to FLAIR into: $REG_DATA"
mkdir -p "$REG_DATA"

# Always link FLAIR as reference/target
if [[ ! -e "$REG_DATA/FLAIR.nii.gz" ]]; then
  ln -sf "$RAW_DATA/FLAIR.nii.gz" "$REG_DATA/FLAIR.nii.gz"
fi

for MOD in T1_RMS T2_caipi; do
  if [[ -f "$REG_DATA/${MOD}.nii.gz" ]]; then
    echo "    $MOD already registered: $REG_DATA/${MOD}.nii.gz"
    continue
  fi

  echo "    Processing $MOD"
  run_fs mri_synthstrip -i "$RAW_DATA/${MOD}.nii.gz" -m "$REG_DATA/${MOD}_brainmask.nii.gz" --gpu

  run_fs mri_coreg \
    --mov "$RAW_DATA/${MOD}.nii.gz" \
    --ref "$REG_DATA/FLAIR.nii.gz" \
    --reg "$REG_DATA/${MOD}_to_FLAIR.lta" \
    --mov-mask "$REG_DATA/${MOD}_brainmask.nii.gz" \
    --ref-mask "$REG_DATA/FLAIR.nii.gz" \
    --threads "$THREADS"

  run_fs mri_vol2vol --cubic \
    --mov "$RAW_DATA/${MOD}.nii.gz" \
    --targ "$REG_DATA/FLAIR.nii.gz" \
    --reg "$REG_DATA/${MOD}_to_FLAIR.lta" \
    --o "$REG_DATA/${MOD}.nii.gz"
done

## resample brain mask
if [[ ! -f "$REG_DATA/mask.nii.gz" ]]; then
  echo "    Resampling brain mask to FLAIR space."
  run_fs mri_vol2vol --nearest \
    --mov "$REG_DATA/T1_RMS_brainmask.nii.gz" \
    --targ "$REG_DATA/FLAIR.nii.gz" \
    --reg "$REG_DATA/T1_RMS_to_FLAIR.lta" \
    --o "$REG_DATA/mask.nii.gz"
else
  echo "    Brain mask already resampled: $REG_DATA/mask.nii.gz"
fi

# can alternatively use a different toolkit for registration (e.g. ANTs) or skip registration if already aligned
# make sure the files look like
# $ tree $REG_DATA
#  ├── FLAIR.nii.gz
#  ├── T1_RMS.nii.gz
#  ├── T1_RMS_brainmask.nii.gz  # optional
#  └── T2_caipi.nii.gz
# and are all in the same space / at the same resolution

############################################
# 5) Run the wrapper
############################################

echo "[5/5] Running YODA wrapper (regression + view aggregation)."

if [[ ! -f "$RUN_DIR/config.yml" ]]; then
  echo "ERROR: RUN_DIR does not contain config.yml: $RUN_DIR" >&2
  exit 2
fi

## wrapper expects -r to run dir, -c relative checkpoint by default
run_yoda_py "$REPO_ROOT/wrapper.py" \
  -r "$RUN_DIR" \
  -c "$CKPT_PATH" \
  -o "$OUT_PRED" \
  "$REG_DATA/T1_RMS.nii.gz" "$REG_DATA/T2_caipi.nii.gz" \
  -m "$REG_DATA/mask.nii.gz" \
  --force  # use the --force, Luke!

echo "Done. Output written to: $OUT_PRED"
