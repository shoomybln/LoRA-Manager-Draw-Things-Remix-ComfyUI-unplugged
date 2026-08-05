#!/bin/bash
# LoRA Manager (Portable Edition) - macOS Launcher
# Starts the standalone server in portable mode and opens the UI.

cd "$(dirname "$0")"

# Force portable mode so the repo-local settings.json is always used.
export LORA_MANAGER_PORTABLE=1

if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

if ! python -c "import aiohttp" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

PORT="${PORT:-8188}"

echo "Starting LoRA Manager (portable) on http://127.0.0.1:${PORT}/loras"
sleep 1
open "http://127.0.0.1:${PORT}/loras"

exec python standalone.py --host 127.0.0.1 --port "${PORT}"
