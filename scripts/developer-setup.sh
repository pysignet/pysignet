#!/bin/bash
set -e

# Install dependencies
poetry install

# Configure git to use the committed hooks in .githooks/
git config core.hooksPath .githooks

echo "Dev environment ready. Run 'poetry shell' or 'source .venv/bin/activate' to activate the virtualenv."
