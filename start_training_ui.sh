#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ -x "$PROJECT_DIR/venv/bin/python" ]]; then
    PYTHON="$PROJECT_DIR/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
else
    echo "[ERROR] Python was not found."
    echo "Install Python or create a project venv, then try again."
    exit 1
fi

export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

echo "Starting Anima ADDifT WebUI at http://127.0.0.1:3001"
exec "$PYTHON" "$PROJECT_DIR/tools/anima_addift_webui.py" --host 127.0.0.1 --port 3001
