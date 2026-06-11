#!/usr/bin/env bash
# Install pyrosm (optional local .osm.pbf reader) without conda.
#
# pyrosm depends on pyrobuf, an unmaintained package whose build breaks on
# modern setuptools ("'PyrobufDistribution' object has no attribute 'dry_run'").
# The fix needs two sequential pip invocations with different flags, which
# requirements.txt cannot express - hence this script.
#
# Usage (after `pip install -r requirements.txt`):
#   bash scripts/install_pyrosm.sh
set -euo pipefail

# 1. Build deps for pyrobuf: an old setuptools plus cython/wheel.
pip install "setuptools<60" wheel cython

# 2. Build pyrobuf against the setuptools above (not an isolated modern one).
pip install --no-build-isolation pyrobuf

# 3. pyrosm itself ships wheels and installs normally once pyrobuf exists.
pip install "pyrosm>=0.6.2"

python -c "import pyrosm; print('pyrosm', pyrosm.__version__, 'OK')"
