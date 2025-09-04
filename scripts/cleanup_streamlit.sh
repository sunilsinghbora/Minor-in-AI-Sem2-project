#!/usr/bin/env bash
# Removes leftover Streamlit .pid and .log files in the repo root
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
shopt -s nullglob
files=("$repo_root"/.streamlit-*.pid "$repo_root"/.streamlit-*.log)
if [ ${#files[@]} -eq 0 ]; then
  echo "No .streamlit-*.pid or .streamlit-*.log files found."
  exit 0
fi
for f in "${files[@]}"; do
  if [ -f "$f" ]; then
    echo "Deleting: $f"
    rm -f -- "$f"
  fi
done
echo "Cleanup complete."
