#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip || true

PKGS=(requests beautifulsoup4 pandas scikit-learn nltk streamlit)
FAILED=0
for pkg in "${PKGS[@]}"; do
  if ! python -m pip install "$pkg"; then
    echo "WARN: Failed to install $pkg (continuing with stdlib fallback)."
    FAILED=1
  fi
done

if [[ "$FAILED" -eq 1 ]]; then
  cat <<MSG
Setup completed with warnings.
Some packages could not be installed, but DelayBot will still run using built-in fallbacks.
MSG
else
  echo "Setup completed successfully."
fi
