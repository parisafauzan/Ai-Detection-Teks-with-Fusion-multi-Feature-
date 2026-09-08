#!/usr/bin/env bash
# Skrip bantu menjalankan backend.
set -e
cd "$(dirname "$0")/backend"
if [ ! -d .venv ]; then
  python -m venv .venv
fi
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
