#!/bin/bash
# Install git hooks for pysignet development
# Run this script after cloning the repository

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
HOOKS_DIR="$PROJECT_ROOT/.git/hooks"

echo "Installing git hooks for pysignet..."

# Create pre-commit hook
cat > "$HOOKS_DIR/pre-commit" << 'EOF'
#!/bin/bash
# Pre-commit hook for pysignet
# Runs quality checks before allowing a commit

set -e  # Exit on first error

echo "Running pre-commit quality checks..."
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track failures
FAILED=0

# 0. Install
echo ""
echo "Installing ..."
poetry install

# 1. Run tests with coverage
echo ""
echo "1. Running tests with coverage check..."
if poetry run python -m pytest tests/ -q --cov=src/pysignet --cov-fail-under=95 2>/dev/null; then
    echo -e "${GREEN}[PASS]${NC} Tests passed with >= 95% coverage"
else
    echo -e "${RED}[FAIL]${NC} Tests failed or coverage below 95%"
    FAILED=1
fi

# 2. Run mypy type checking
echo ""
echo "2. Running mypy type checking..."
if poetry run python -m mypy src/ --ignore-missing-imports --no-error-summary 2>/dev/null; then
    echo -e "${GREEN}[PASS]${NC} Type checking passed"
else
    echo -e "${RED}[FAIL]${NC} Type checking failed"
    FAILED=1
fi

# 3. Check for syntax errors in examples
echo ""
echo "3. Checking example syntax..."
EXAMPLE_ERRORS=0
for f in examples/*.py; do
    if ! python -m py_compile "$f" 2>/dev/null; then
        echo -e "${RED}[FAIL]${NC} Syntax error in $f"
        EXAMPLE_ERRORS=1
    fi
done
if [ $EXAMPLE_ERRORS -eq 0 ]; then
    echo -e "${GREEN}[PASS]${NC} All examples have valid syntax"
else
    FAILED=1
fi

echo ""
echo "======================================"

if [ $FAILED -eq 1 ]; then
    echo -e "${RED}Pre-commit checks FAILED${NC}"
    echo "Fix the issues above before committing."
    exit 1
else
    echo -e "${GREEN}All pre-commit checks PASSED${NC}"
    exit 0
fi
EOF

chmod +x "$HOOKS_DIR/pre-commit"

echo "Git hooks installed successfully!"
echo ""
echo "The pre-commit hook will run before each commit to check:"
echo "  - All tests pass"
echo "  - Coverage >= 95%"
echo "  - mypy type checking passes"
echo "  - Examples have valid syntax"
echo ""
echo "To skip hooks temporarily, use: git commit --no-verify"
