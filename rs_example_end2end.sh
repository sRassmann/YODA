#!/usr/bin/env bash
set -euo pipefail

# RS example end-to-end pipeline for YODA
# - downloads RS example ZIP from Zenodo
# - (optionally) fetches YODA weights (run dir) from Zenodo
# - sets up runtime via singularity OR docker OR local python env
# - registers T1/T2 to FLAIR via FreeSurfer tools
# - runs YODA wrapper on the registered images
#
# Requirements (depending on mode):
#   - FreeSurfer license: FS_LICENSE must point to a valid license file (not required for single-modality or pre-registered inputs)
#   - singularity/apptainer OR docker OR a local python env (conda/venv) with YODA deps + FreeSurfer installed
#   - NVIDIA GPU + drivers if you want GPU execution (recommended)

############################################
# 0) User-controlled paths via env vars
############################################

# Which runtime to use for *YODA inference*.
#   singularity: use dagobah.sif downloaded from Zenodo
#   docker:      run srassmann/dif:latest as a container (renamed for the joke)
#   local:       use system python (assumes deps installed, e.g. via conda)
export RUNTIME="${RUNTIME:-singularity}"

# Where intermediate/extracted data and outputs go
export PROCESSING_PATH="${PROCESSING_PATH:-"output"}"

# Path to YODA run directory (must contain config.yml and ckpt/last.pth unless overridden).
# If not set by the user, we download default RS weights from Zenodo.
export RUN_DIR="${RUN_DIR:-}"

# Where singularity .sif images are stored
export SINGULARITY_DIR="${SINGULARITY_DIR:-$HOME/singularity}"

# Optional: checkpoint path (relative to RUN_DIR or absolute)
export CKPT_PATH="${CKPT_PATH:-ckpt/last.pth}"

# Optional: output prediction path
export OUT_PRED="${OUT_PRED:-$PROCESSING_PATH/output/pred.nii.gz}"

# Optional: execution toggles
export USE_GPU="${USE_GPU:-1}"   # 1 => enable GPU (singularity: --nv, docker: --gpus all if available)
export THREADS="${THREADS:-16}"

# Optional: set this to 0 if you don't want to download images/weights (assumes they already exist)
export DOWNLOAD_ASSETS="${DOWNLOAD_ASSETS:-1}"

# Optional: docker image names
export YODA_DOCKER_IMAGE="${YODA_DOCKER_IMAGE:-srassmann/dif:latest}"
export YODA_DOCKER_NAME="${YODA_DOCKER_NAME:-dagobah}"
export FS_DOCKER_IMAGE="${FS_DOCKER_IMAGE:-freesurfer/freesurfer:7.4.1}"

############################################
# URLs (Zenodo)
############################################

ZENODO_YODA="https://zenodo.org/records/19088324"
YODA_SIF_URL="$ZENODO_YODA/files/dagobah.sif"
RS_RUN_ZIP_URL="$ZENODO_YODA/files/rs_FLAIR_from_T1T2.zip"  # default, see zenodo for more
RS_RAW_ZIP_URL="https://zenodo.org/records/19133592/files/sub_rs_mri_struc_only.zip"

############################################
# Derived locations
############################################

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Singularity bind mounts require absolute host paths.
abspath() {
  local p="$1"
  if [[ -z "$p" ]]; then
    echo ""
    return 0
  fi
  if [[ "$p" = /* ]]; then
    echo "$p"
    return 0
  fi
  # Use REPO_ROOT as the anchor for relative paths so the script behaves consistently
  # regardless of current working directory.
  echo "$(cd "$REPO_ROOT" && realpath -m "$p")"
}

PROCESSING_PATH="$(abspath "$PROCESSING_PATH")"
SINGULARITY_DIR="$(abspath "$SINGULARITY_DIR")"
OUT_PRED="$(abspath "$OUT_PRED")"

# Don't absolutize RUN_DIR until after auto-download (it may be empty at this point).

RAW_DATA="$PROCESSING_PATH/data/rs_example_raw/subj_0000"
REG_DATA="$PROCESSING_PATH/data/rs_example_registered/subj_0000"

mkdir -p "$PROCESSING_PATH" "$SINGULARITY_DIR" \
  "$(dirname "$RAW_DATA")" "$(dirname "$REG_DATA")" "$(dirname "$OUT_PRED")"

YODA_SIF="$SINGULARITY_DIR/dagobah.sif"
FS_SIF="$SINGULARITY_DIR/freesurfer_74.sif"

# Binds: repo (for wrapper + python code), processing path (data + outputs), and run dir (weights/config)
# Note: binds differ slightly between runtimes; these are the canonical host paths.

############################################
# Helpers
############################################

die() { echo "ERROR: $*" >&2; exit 2; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

############################################
# Runtime functions (YODA)
############################################

run_yoda_py() {
  local script="$1"; shift

  case "$RUNTIME" in
    singularity)
      need_cmd singularity
      local -a nv_args=()
      if [[ "$USE_GPU" == "1" ]]; then nv_args+=("--nv"); fi

      # Bind: repo, processing, run dir
      local -a binds=(
        "-B" "$REPO_ROOT:$REPO_ROOT"
        "-B" "$PROCESSING_PATH:$PROCESSING_PATH"
        "-B" "$RUN_DIR:$RUN_DIR:ro"
      )

      singularity exec "${nv_args[@]}" "${binds[@]}" "$YODA_SIF" python "$script" "$@"
      ;;

    docker)
      need_cmd docker

      # We run "docker run --rm" each time
      local -a gpu_args=()
      if [[ "$USE_GPU" == "1" ]]; then
        # Works on recent docker/nvidia-container-toolkit; if unavailable, docker will error.
        gpu_args+=("--gpus" "all")
      fi

      docker run --rm --name "${YODA_DOCKER_NAME}" \
        "${gpu_args[@]}" \
        -v "$REPO_ROOT:$REPO_ROOT" \
        -v "$PROCESSING_PATH:$PROCESSING_PATH" \
        -v "$RUN_DIR:$RUN_DIR:ro" \
        -w "$REPO_ROOT" \
        "$YODA_DOCKER_IMAGE" \
        python "$script" "$@"
      ;;

    local)
      # Assumes user has activated env externally (e.g. conda activate ...)
      python "$script" "$@"
      ;;

    *)
      die "Unknown RUNTIME='$RUNTIME' (expected: singularity|docker|local)"
      ;;
  esac
}

############################################
# Runtime functions (FreeSurfer registration)
############################################

run_fs() {
  case "$RUNTIME" in
    singularity)
      need_cmd singularity
      local -a nv_args=()
      if [[ "$USE_GPU" == "1" ]]; then nv_args+=("--nv"); fi

      [[ -n "${FS_LICENSE:-}" ]] || die "FS_LICENSE is not set (required by FreeSurfer)"
      [[ -f "${FS_LICENSE}" ]] || die "FS_LICENSE does not exist: ${FS_LICENSE}"

      local -a binds=(
        "-B" "$REPO_ROOT:$REPO_ROOT"
        "-B" "$PROCESSING_PATH:$PROCESSING_PATH"
        "-B" "$RUN_DIR:$RUN_DIR:ro"
        "-B" "$FS_LICENSE:$FS_LICENSE:ro"
      )

      singularity exec "${nv_args[@]}" "${binds[@]}" \
        --env SUBJECTS_DIR="$PROCESSING_PATH/freesurfer_subjects" \
        --env FS_LICENSE="$FS_LICENSE" \
        "$FS_SIF" "$@"
      ;;

    docker)
      need_cmd docker
      [[ -n "${FS_LICENSE:-}" ]] || die "FS_LICENSE is not set (required by FreeSurfer)"
      [[ -f "${FS_LICENSE}" ]] || die "FS_LICENSE does not exist: ${FS_LICENSE}"

      local -a gpu_args=()
      if [[ "$USE_GPU" == "1" ]]; then gpu_args+=("--gpus" "all"); fi

      docker run --rm \
        "${gpu_args[@]}" \
        -v "$PROCESSING_PATH:$PROCESSING_PATH" \
        -v "$REPO_ROOT:$REPO_ROOT" \
        -v "$FS_LICENSE:$FS_LICENSE:ro" \
        -e SUBJECTS_DIR="$PROCESSING_PATH/freesurfer_subjects" \
        -e FS_LICENSE="$FS_LICENSE" \
        -w "$REPO_ROOT" \
        "$FS_DOCKER_IMAGE" \
        "$@"
      ;;

    local)
      # If user has a local FreeSurfer install, they can use it here.
      "$@"
      ;;

    *)
      die "Unknown RUNTIME='$RUNTIME' (expected: singularity|docker|local)"
      ;;
  esac
}

############################################
# 1) Download RS example data
############################################

need_cmd wget
need_cmd unzip

ZIP_DIR="$PROCESSING_PATH/downloads"
ZIP_FILE="$ZIP_DIR/sub_rs_mri_raw.zip"
mkdir -p "$ZIP_DIR"

if [[ "$DOWNLOAD_ASSETS" == "1" && ! -f "$ZIP_FILE" ]]; then
  echo "[1/6] Downloading RS example ZIP to: $ZIP_FILE"
  wget -O "$ZIP_FILE" "$RS_RAW_ZIP_URL"
else
  echo "[1/6] RS raw ZIP already present (or DOWNLOAD_ASSETS=0): $ZIP_FILE"
fi

############################################
# 2) Unzip (extract only required files)
############################################

echo "[2/6] Extracting T1/T2/FLAIR into: $RAW_DATA"
mkdir -p "$RAW_DATA"

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
# 3) Fetch YODA assets (SIF + weights)
############################################

echo "[3/6] Preparing runtime assets (SIF/weights) for RUNTIME='$RUNTIME'."

if [[ -z "$RUN_DIR" ]]; then
  # Default: download weights + config bundle from Zenodo.
  DEFAULT_RUN_BASE="$PROCESSING_PATH/runs"
  DEFAULT_RUN_NAME="rs_FLAIR_from_T1T2"
  DEFAULT_RUN_ZIP="$ZIP_DIR/${DEFAULT_RUN_NAME}.zip"
  DEFAULT_RUN_DIR="$DEFAULT_RUN_BASE/$DEFAULT_RUN_NAME"

  mkdir -p "$DEFAULT_RUN_BASE"

  if [[ "$DOWNLOAD_ASSETS" == "1" && ! -f "$DEFAULT_RUN_ZIP" && ! -d "$DEFAULT_RUN_DIR" ]]; then
    echo "    Downloading default run ZIP (weights/config) from Zenodo: $DEFAULT_RUN_ZIP"
    wget -O "$DEFAULT_RUN_ZIP" "$RS_RUN_ZIP_URL"
  fi

  if [[ -d "$DEFAULT_RUN_DIR" ]]; then
    echo "    Default run dir already present: $DEFAULT_RUN_DIR"
  else
    echo "    Extracting default run dir into: $DEFAULT_RUN_BASE"
    unzip -o "$DEFAULT_RUN_ZIP" -d "$DEFAULT_RUN_BASE"
  fi

  RUN_DIR="$DEFAULT_RUN_DIR"
  export RUN_DIR
fi

RUN_DIR="$(abspath "$RUN_DIR")"
export RUN_DIR

[[ -f "$RUN_DIR/config.yml" ]] || die "RUN_DIR must contain config.yml: $RUN_DIR"

if [[ "$RUNTIME" == "singularity" ]]; then
  need_cmd singularity

  if [[ "$DOWNLOAD_ASSETS" == "1" && ! -f "$YODA_SIF" ]]; then
    echo "    Downloading YODA singularity image (dagobah.sif) from Zenodo: $YODA_SIF"
    wget -O "$YODA_SIF" "$YODA_SIF_URL"
  fi
  [[ -f "$YODA_SIF" ]] || die "Missing YODA SIF: $YODA_SIF"

  # FreeSurfer image: build from dockerhub due to license constraints
  if [[ "$DOWNLOAD_ASSETS" == "1" && ! -f "$FS_SIF" ]]; then
    echo "    Building FreeSurfer SIF from docker://freesurfer/freesurfer:latest => $FS_SIF"
    singularity build "$FS_SIF" docker://freesurfer/freesurfer:7.4.1
  fi
  [[ -f "$FS_SIF" ]] || die "Missing FreeSurfer SIF: $FS_SIF"
fi

############################################
# 4) Register using FreeSurfer
############################################

echo "[4/6] Registering T1/T2 to FLAIR into: $REG_DATA"
mkdir -p "$REG_DATA"

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

echo "[5/6] Running YODA wrapper (regression + view aggregation)."

echo "    RUNTIME=$RUNTIME"
echo "    RUN_DIR=$RUN_DIR"

demo_wrapper="$REPO_ROOT/wrapper.py"
[[ -f "$demo_wrapper" ]] || die "wrapper.py not found at: $demo_wrapper"

run_yoda_py "$demo_wrapper" \
  -r "$RUN_DIR" \
  -c "$CKPT_PATH" \
  -o "$OUT_PRED" \
  "$REG_DATA/T1_RMS.nii.gz" "$REG_DATA/T2_caipi.nii.gz" \
  -m "$REG_DATA/mask.nii.gz" \
  --force  # use the --force, Luke!

echo "[6/6] Done. Output written to: $OUT_PRED"
