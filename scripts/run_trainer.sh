#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

TK_SILENCE_DEPRECATION=1 PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m cad_ai_suite.apps.trainer_app
