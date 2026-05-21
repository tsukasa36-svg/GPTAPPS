#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "Python environment not found."
  echo "Run this first in Terminal:"
  echo "python3 -m venv .venv"
  echo "source .venv/bin/activate"
  echo "pip install cadquery"
  read -r -p "Press Enter to close..."
  exit 1
fi

source .venv/bin/activate

echo "Starting CAD AI Workbench..."
echo "A browser window should open at http://127.0.0.1:8765"
echo "Keep this terminal window open while using the app."
echo ""
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m cad_ai_suite.apps.web_workbench
