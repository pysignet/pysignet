# Contributing to pysignet

Thank you for your interest in contributing to pysignet! This document provides
guidelines and instructions for contributing.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Code Quality Checks](#code-quality-checks)

## Getting Started

This project uses [Poetry](https://python-poetry.org/) for dependency management
and packaging. You will need Poetry installed to contribute.

### Prerequisites

- **Python 3.12 or higher**
- **Poetry** - Install via the official installer: ```bash curl -sSL
  https://install.python-poetry.org | python3 - ``` Or see [Poetry installation
  docs](https://python-poetry.org/docs/#installation) for other methods.
- **Git**

### Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/pysignet.git
   cd pysignet
   ```

2. **Install dependencies with Poetry:** ```bash poetry install ``` This creates
   a virtual environment and installs all dependencies including dev tools
   (pytest, mypy, pylint, black).

3. **Activate the virtual environment:**
   ```bash
   poetry shell
   ```

4. **Verify setup by running tests:**
   ```bash
   poetry run python -m pytest tests/
   ```

### Running Commands

All commands should be run through Poetry to ensure the correct environment:

```bash
# Run tests
poetry run python -m pytest tests/

# Run type checking
poetry run mypy src/

# Run linting
poetry run pylint src/

# Format code
poetry run black src/ tests/ examples/
```

Alternatively, activate the shell first with `poetry shell`, then run commands
directly.

## Development Workflow

This project strictly follows **Test-Driven Development (TDD)**. All new
features and bug fixes must follow the RED-GREEN-REFACTOR cycle.

### TDD Workflow

#### 1. RED: Write Failing Tests First

Before writing any implementation code, write tests that define the expected
behavior:

```bash
# Create or edit test file
emacs tests/test_new_feature.py

# Run tests - they should FAIL
pytest tests/test_new_feature.py -v
```

Tests should cover:
- **Happy path**: Normal, expected usage
- **Edge cases**: Empty inputs, boundary values, extreme sizes
- **Error conditions**: Invalid inputs, type mismatches
- **Gradient flow**: For differentiable operations

#### 2. GREEN: Write Minimum Implementation

Write only the code needed to make tests pass:

```bash
# Implement the feature
emacs src/pysignet/module.py

# Run tests - they should PASS
pytest tests/test_new_feature.py -v
```

#### 3. REFACTOR: Clean Up

Improve code quality while keeping tests passing:

- Apply Google Python Style Guide
- Add comprehensive docstrings
- Add type hints
- Run full test suite after each change

```bash
pytest tests/ -v
```

### Commit Guidelines

Use [Conventional Commits](https://www.conventionalcommits.org/) format for all
new contributions:

- `feat: add new feature description`
- `fix: correct bug description`
- `test: add tests for feature`
- `refactor: restructure code without changing behavior`
- `docs: update documentation`
- `style: formatting changes (no code change)`

Examples:
```bash
git commit -m "feat: add Lukasiewicz t-norm implementation"
git commit -m "test: add comprehensive edge case tests for ForAll"
git commit -m "fix: handle empty batch in domain quantifier"
```

> Note: Some historical commits predate these guidelines.

## Code Style

This project follows the [Google Python Style
Guide](https://google.github.io/styleguide/pyguide.html).

### Key Guidelines

| Aspect              | Guideline             |
|---------------------|-----------------------|
| Line length         | Maximum 80 characters |
| Indentation         | 4 spaces              |
| Functions/variables | `snake_case`          |
| Classes             | `PascalCase`          |
| Constants           | `UPPER_CASE`          |
| Private attributes  | `_leading_underscore` |

### Type Hints

Type hints are **required** for all function signatures. The project uses mypy
in strict mode.

```python
# Good
def compute_loss(
    expression: sp.Basic,
    predicates: dict[str, Callable[..., torch.Tensor]],
    reduction: str = "mean"
) -> torch.Tensor:
    ...

# Bad - missing type hints
def compute_loss(expression, predicates, reduction="mean"):
    ...
```

Use modern Python typing syntax:
- Use `X | Y` instead of `Union[X, Y]`
- Use `list[X]` instead of `List[X]`
- Use `dict[K, V]` instead of `Dict[K, V]`

### Docstrings

Use Google-style docstrings for public modules, classes, and functions:

```python
def compile_logic(
    expression: sp.Basic,
    predicates: Dict[str, Predicate | Callable[..., torch.Tensor]],
    tnorm: Optional[TNorm] = None,
) -> CompiledExpression:
    """Compile a logical expression into a differentiable function.

    Args:
        expression: A SymPy expression representing the logical constraint.
        predicates: Mapping from symbol names to predicate functions.
        tnorm: T-norm instance for relaxation (default: MixedTNorm).

    Returns:
        A compiled expression that can be evaluated on tensor inputs.

    Raises:
        ValueError: If an unknown symbol is found in the expression.
        TypeError: If predicates have incorrect signatures.

    Example:
        ```python
        P, Q = Symbol("P Q")
        X = Variable("X")
        compiled = compile_logic(And(P(X), Q(X)), {"P": model_p, "Q": model_q})
        result = compiled(X=input_tensor)
        ```
    """
```

### Output Formatting

- **No non-ASCII unicode** in code, comments, or documentation unless absolutely
  necessary. For example:
  - Use `->` not arrows
  - Use `forall` not mathematical symbols
  - Spell out Greek letters in prose

## Testing

### Running Tests

```bash
# Run all tests with coverage
poetry run python -m pytest tests/

# Run specific test file
poetry run python -m pytest tests/test_tnorms.py -v

# Run specific test function
poetry run python -m pytest tests/test_tnorms.py::test_product_and -v

# Run tests matching a pattern
poetry run python -m pytest tests/ -k "gradient" -v

# Run without coverage (faster)
poetry run python -m pytest tests/ --no-cov
```

### Coverage Requirements

- Minimum **95% code coverage** is enforced
- Coverage reports are generated automatically
- View HTML report: `open htmlcov/index.html`

### Test Organization

Tests mirror the source structure:

```
tests/
    compilation/
        test_base.py
        test_tnorm_compiler.py
    logic/
        test_symbols.py
        test_quantifiers.py
    tnorms/
        test_product.py
        test_lukasiewicz.py
        test_godel.py
    test_api.py
    test_loss.py
    conftest.py          # Shared fixtures
```

### Writing Good Tests

Use pytest fixtures and parametrize for clean, comprehensive tests:

```python
import pytest
import torch

@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.tensor([0.3, 0.7, 0.5])

@pytest.mark.parametrize("tnorm_name,expected", [
    ("rproduct", 0.105),
    ("lukasiewicz", 0.0),
    ("godel", 0.3),
])
def test_conjunction_tnorms(tnorm_name, expected, sample_tensor):
    """Test AND operation across different t-norms."""
    tnorm = get_tnorm(tnorm_name)
    result = tnorm.conjunction(sample_tensor)
    assert torch.isclose(result, torch.tensor(expected), atol=1e-3)
```

## Pull Request Process

### Before Submitting

1. **Ensure all tests pass:**
   ```bash
   poetry run python -m pytest tests/
   ```

2. **Run type checking:**
   ```bash
   poetry run python -m mypy src/ --ignore-missing-imports
   ```

3. **Run linting:**
   ```bash
   poetry run pylint src/
   ```

4. **Format code:**
   ```bash
   black src/ tests/ examples/
   ```

### PR Checklist

- [ ] Tests written for new functionality (TDD)
- [ ] All tests pass locally
- [ ] Type hints added for all new functions
- [ ] Docstrings added for public APIs
- [ ] No mypy errors or warnings
- [ ] Pylint passes clean (`poetry run pylint src/`)
- [ ] Coverage remains above 95%
- [ ] Commit messages follow conventional format
- [ ] PR description explains the changes

### PR Description Template

```markdown
## Summary
Brief description of changes.

## Changes
- Change 1
- Change 2

## Testing
How was this tested?

## Checklist
- [ ] Tests added
- [ ] Documentation updated
- [ ] Type hints complete
```

## Code Quality Checks

### All Checks at Once

```bash
# Run all quality checks
poetry run python -m pytest tests/                              # Tests + coverage
poetry run python -m mypy src/ --ignore-missing-imports        # Type checking
poetry run pylint src/                                         # Linting
poetry run black --check src/ tests/ examples/                 # Formatting check
```

### Individual Checks

| Check         | Command              | Requirement       |
|---------------|----------------------|-------------------|
| Tests         | `pytest tests/`      | All pass          |
| Coverage      | `pytest --cov`       | >= 95%            |
| Type checking | `poetry run python -m mypy src/ --ignore-missing-imports` | Zero errors |
| Linting       | `poetry run pylint src/`                                  | Score 10.00 |
| Formatting    | `black --check src/` | No changes needed |

### Pre-commit Hook

The repository includes a pre-commit hook in `.githooks/pre-commit`. It is
already activated in this repo via `git config core.hooksPath .githooks`.

The hook runs the same checks as CI, in order:
1. `pytest tests/ -q --cov=src/pysignet --cov-fail-under=95`
2. `mypy src/ --ignore-missing-imports`
3. `pylint src/`
4. Syntax check of all `examples/*.py`

To run it manually:
```bash
.githooks/pre-commit
```

The hook must stay in sync with `.github/workflows/ci.yml`. If CI adds a new
check, add it to the hook as well.

## Questions?

- Open an issue for bugs or feature requests
- Check existing issues before creating new ones
- For questions about usage, see the documentation and examples

## License

By contributing, you agree that your contributions will be licensed under the
MIT License.
