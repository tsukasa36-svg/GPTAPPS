#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m cad_ai_suite.apps.web_workbench
