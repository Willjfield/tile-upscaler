#!/usr/bin/env bash
# Install pyrosm (optional local .osm.pbf reader) without conda.
#
# pyrobuf (pyrosm's dependency) is unmaintained and breaks in two ways on modern
# Python (3.12+):
#   1. setuptools>=60 removed Distribution.dry_run (PyrobufDistribution crashes)
#   2. pip's PEP 517 metadata step + setuptools<60 cannot import build_meta
#
# Fix: keep setuptools>=70 (distutils shim + build_meta on 3.12), download the
# pyrobuf sdist, patch PyrobufDistribution.dry_run, and build with
# --no-build-isolation.
#
# Usage (after `pip install -r requirements.txt`):
#   bash scripts/install_pyrosm.sh
set -euo pipefail

PYROBUF_VERSION="${PYROBUF_VERSION:-0.9.3}"
PYROBUF_URL="https://files.pythonhosted.org/packages/source/p/pyrobuf/pyrobuf-${PYROBUF_VERSION}.tar.gz"

log() { printf '==> %s\n' "$*"; }

if python -c "import pyrosm" 2>/dev/null; then
  python -c "import pyrosm; print('pyrosm already installed:', pyrosm.__version__)"
  exit 0
fi

log "Installing pyrobuf build dependencies"
pip install "setuptools>=70" wheel cython jinja2

tmpdir="$(mktemp -d)"
cleanup() { rm -rf "$tmpdir"; }
trap cleanup EXIT

log "Downloading pyrobuf ${PYROBUF_VERSION}"
curl -fsSL "$PYROBUF_URL" | tar -xz -C "$tmpdir"
src="$(echo "$tmpdir"/pyrobuf-"${PYROBUF_VERSION}")"
if [[ ! -d "$src" ]]; then
  src="$(find "$tmpdir" -maxdepth 1 -type d -name 'pyrobuf-*' | head -1)"
fi
if [[ ! -f "$src/setup.py" ]]; then
  echo "pyrobuf source not found under $tmpdir" >&2
  exit 1
fi

log "Patching pyrobuf for setuptools>=60 (dry_run)"
python - "$src" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "setup.py"
text = path.read_text()
needle = "class PyrobufDistribution(Distribution):"
patch = "class PyrobufDistribution(Distribution):\n    dry_run = False"
if "dry_run = False" not in text:
    if needle not in text:
        raise SystemExit(f"patch point not found in {path}")
    path.write_text(text.replace(needle, patch, 1))
    print("patched", path)
else:
    print("already patched", path)
PY

log "Building and installing pyrobuf"
pip install --no-build-isolation "$src"
python -c "import pyrobuf; print('pyrobuf OK')"

log "Installing pyrosm"
pip install "pyrosm>=0.6.2"
python -c "import pyrosm; print('pyrosm', pyrosm.__version__, 'OK')"
