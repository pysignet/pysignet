#!/bin/bash
# Configure git to use the committed hooks in .githooks/
git config core.hooksPath .githooks
echo "Git hooks enabled. Pre-commit checks will run before each commit."
