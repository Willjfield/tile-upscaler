#!/usr/bin/env bash
# Bootstrap a RunPod (or other cloud GPU) pod against a persisted network volume.
#
# Python virtualenvs are NOT portable across container images: .venv/bin/python is
# a symlink to the system interpreter path from the pod where the venv was created.
# On a new pod you will see "No such file or directory" even though .venv/ exists.
# This script recreates the venv and reinstalls dependencies.
#
# Usage (from repo root, after SSH into a fresh pod):
#   bash scripts/runpod_setup.sh              # recreate venv only if broken
#   bash scripts/runpod_setup.sh --recreate   # force fresh venv (recommended on new pods)
#   bash scripts/runpod_setup.sh --skip-pyrosm  # skip optional pyrosm install
#
# Environment overrides:
#   TORCH_INDEX_URL   PyTorch wheel index (default: inferred from nvidia-smi, else cu124)
#   HF_HOME           Hugging Face cache dir (default: <repo>/.cache/huggingface)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RECREATE=0
SKIP_PYROSM=0
for arg in "$@"; do
  case "$arg" in
    --recreate) RECREATE=1 ;;
    --skip-pyrosm) SKIP_PYROSM=1 ;;
    -h|--help)
      sed -n '2,16p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg (try --help)" >&2
      exit 1
      ;;
  esac
done

log() { printf '==> %s\n' "$*"; }

venv_python_ok() {
  [[ -x .venv/bin/python ]] || return 1
  .venv/bin/python -c "import sys; print(sys.executable)" >/dev/null 2>&1
}

infer_torch_index() {
  if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    echo "$TORCH_INDEX_URL"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local ver
    ver="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
    case "$ver" in
      12.8|12.7|12.6|12.5|12.4) echo "https://download.pytorch.org/whl/cu124" ;;
      12.3|12.2|12.1) echo "https://download.pytorch.org/whl/cu121" ;;
      12.0) echo "https://download.pytorch.org/whl/cu121" ;;
      11.8) echo "https://download.pytorch.org/whl/cu118" ;;
      *)
        log "Unmapped CUDA Version $ver from nvidia-smi; defaulting to cu124"
        echo "https://download.pytorch.org/whl/cu124"
        ;;
    esac
    return
  fi
  log "nvidia-smi not found; defaulting torch index to cu124"
  echo "https://download.pytorch.org/whl/cu124"
}

setup_hf_cache() {
  local cache="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
  export HF_HOME="$cache"
  mkdir -p "$HF_HOME"
  local marker="# tile-upscaler HF_HOME"
  if [[ -f "$HOME/.bashrc" ]] && ! grep -qF "$marker" "$HOME/.bashrc" 2>/dev/null; then
    {
      echo ""
      echo "$marker"
      echo "export HF_HOME=\"$HF_HOME\""
    } >> "$HOME/.bashrc"
    log "Appended HF_HOME=$HF_HOME to ~/.bashrc"
  fi
}

patch_basicsr() {
  .venv/bin/python - <<'PY'
import importlib.util
import pathlib
import sys

spec = importlib.util.find_spec("basicsr")
if spec is None or spec.origin is None:
    print("basicsr not installed; skip patch")
    sys.exit(0)
path = pathlib.Path(spec.origin).parent / "data" / "degradations.py"
if not path.is_file():
    print("basicsr degradations.py not found; skip patch")
    sys.exit(0)
text = path.read_text()
old = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
new = "from torchvision.transforms.functional import rgb_to_grayscale"
if old not in text:
    if new in text:
        print("basicsr already patched:", path)
    else:
        print("basicsr patch string not found; skip:", path)
    sys.exit(0)
path.write_text(text.replace(old, new))
print("patched basicsr:", path)
PY
}

log "Repo: $REPO_ROOT"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1 || true
  nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: /Driver CUDA: /p' | head -1 || true
else
  log "WARNING: nvidia-smi not found — GPU may not be visible yet"
fi

TORCH_INDEX="$(infer_torch_index)"
log "PyTorch index: $TORCH_INDEX"

setup_hf_cache

if [[ "$RECREATE" -eq 1 ]]; then
  log "Removing existing .venv (--recreate)"
  rm -rf .venv
elif venv_python_ok; then
  log "Existing .venv looks usable (run with --recreate to rebuild on a new pod image)"
else
  log "Existing .venv missing or broken (symlink target from old pod?) — recreating"
  rm -rf .venv
fi

if ! venv_python_ok; then
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found on PATH" >&2
    exit 1
  fi
  log "Creating venv with $(python3 --version 2>&1) at $(command -v python3)"
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

log "Upgrading pip"
python -m pip install -U pip wheel

log "Installing torch + torchvision"
pip install torch torchvision --index-url "$TORCH_INDEX"

log "Installing requirements.txt"
pip install -r requirements.txt

if [[ "$SKIP_PYROSM" -eq 0 ]]; then
  log "Installing pyrosm (local .osm.pbf reader)"
  bash scripts/install_pyrosm.sh
else
  log "Skipping pyrosm (--skip-pyrosm)"
fi

log "Patching basicsr for torchvision>=0.17"
patch_basicsr

log "Sanity checks"
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
python -c "import diffusers; print('diffusers', diffusers.__version__)"
python -c "
try:
    import pyrosm
    print('pyrosm', pyrosm.__version__)
except ImportError:
    print('pyrosm not installed (Overpass fallback still works)')
"

cat <<EOF

Done. Next steps:

  cd $REPO_ROOT
  source .venv/bin/activate

  # pull latest code if needed
  git pull

  # smoke test
  python run_experiment.py --limit 5

  # full run
  python run_experiment.py

HF cache: \$HF_HOME
On the next new pod: bash scripts/runpod_setup.sh --recreate
EOF
