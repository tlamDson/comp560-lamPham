#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${REPO_ROOT}/run.sh"

if [[ ! -x "${RUNNER}" ]]; then
  echo "ERROR: ${RUNNER} is not executable. Run: chmod +x ${RUNNER}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

echo ">>> Compiling requirements-lock.txt from layered manifests..."
"${RUNNER}" -m piptools compile \
  --resolver=backtracking \
  --allow-unsafe \
  --output-file requirements-lock.txt \
  requirements-core.txt requirements-phi.txt

echo ">>> Generating conda-lock.yml from environment.yml..."
"${RUNNER}" -m conda_lock lock \
  --file environment.yml \
  --platform linux-64 \
  --lockfile conda-lock.yml

echo ">>> Lock generation complete"
echo "    - ${REPO_ROOT}/requirements-lock.txt"
echo "    - ${REPO_ROOT}/conda-lock.yml"