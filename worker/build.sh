#!/bin/sh
# Builds the Fuzzy Worker for this Mac or Linux computer into dist/ (PyInstaller can't cross-build).
# Needs Python 3.12 with tkinter (macOS: python.org installer; Debian/Ubuntu: apt install python3-tk).
set -e
cd "$(dirname "$0")"
python3 -m venv .buildenv
. .buildenv/bin/activate
pip install --quiet -r requirements.txt pyinstaller
pyinstaller --noconfirm --onefile --windowed --name FuzzyWorker --paths . \
    --collect-submodules fuzzy_worker fuzzy_worker/__main__.py
echo "Built: $(pwd)/dist"
