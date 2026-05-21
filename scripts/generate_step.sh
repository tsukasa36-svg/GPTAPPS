#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

if [ "$#" -eq 0 ]; then
  prompt="Generate an aluminum L bracket 80mm by 120mm, 6mm thick, with four M6 holes and a triangular gusset"
else
  prompt="$*"
fi

PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m cad_ai_suite.apps.generator_app "$prompt"
