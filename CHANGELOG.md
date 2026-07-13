# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Version boundaries below follow the `version` field in `pyproject.toml`
(0.1.0 -> 0.2.0 -> 1.0.0); entries are grouped by theme within each range
rather than listing every individual commit.

## [Unreleased]

- v1.0.0 has not been tagged/published yet. Remaining step: push the
  `v1.0.0` git tag to trigger the automated PyPI publish workflow.

## [1.0.0] - 2026-06-23 to 2026-07-12

### Added

- PyPI release infrastructure: package metadata (description, keywords,
  classifiers, project URLs) in `pyproject.toml`, and a `publish.yml`
  GitHub Actions workflow using OIDC Trusted Publishers (no token secret
  required), triggered on `v*` tags (2026-06-23)
- `ruff` added to CI linting, alongside pylint and mypy (2026-07-11)

### Changed

- `examples/` removed from version control; dropped its syntax-check step
  from the pre-commit hook accordingly (2026-06-24)
- Quickstart docs/README simplified to pass `nn.Module` predicates
  directly, without manual `squeeze` wrappers (2026-06-23)
- Notebook content refresh and a repo-wide linting cleanup pass
  (2026-07-12)

## [0.2.0] - 2025-12-29 to 2026-06-23

`pyproject.toml` stayed pinned at 0.2.0 for this entire six-month span
(no tag or publish ever happened at 0.2.0 - the version was bumped
straight to 1.0.0 once release infrastructure was in place). Package
renamed `logic_as_loss` -> `pysignet` at the start of this range
(2025-12-29). This range covers the bulk of development: First-Order
Logic support, mixed t-norms, the documentation site, and CI hardening.

### Added - First-Order Logic (FOL)

- `Variable` system and predicate application over variables/constants,
  e.g. `Digit(X)`, `Digit(X, 5)` (2026-01-02)
- Automatic universal quantification of free variables over the batch
  dimension, plus explicit `ForAll`/`Exists` quantifiers over finite
  domains (2026-01-02 to 2026-01-03)
- N-ary predicates, partial variable binding (`.partial(**bindings)`),
  and constant arguments mixed with variables in predicates
  (2026-01-03 to 2026-01-13)
- `quantify` parameter (`'forall'` / `'exists'` / `'none'`) and
  keyword-argument variable binding (`compiled(X=x)`); batch reduction
  refactored into a reusable mixin (2026-01-18)
- `fol/` module renamed to `logic/` for a less jargon-heavy public API
  (2026-01-02)
- LTU (Logic Tensor Unit) compiler added as an alternative compilation
  strategy, with a configurable sigmoid sharpness multiplier, later
  exposed and documented on the public API (2026-01-03, 2026-06-15)

### Added - Mixed T-Norms

- `MixedTNorm`: Godel for large arities, product-based t-norm otherwise;
  made the default compiler t-norm (2026-02-02 to 2026-03-05)
- Log-space computation for numerical stability of product t-norms on
  large batches (2026-02-11)

### Changed - Architecture

- `LogicLoss` renamed to `LogicCompiler`, then compilation was separated
  from loss computation entirely: `TNormCompiler` produces a
  differentiable closure, `LogicLoss` wraps it for loss computation;
  `compile_logic()` / `logic_to_loss()` factory functions added
  (2025-12-28 to 2025-12-30, refactored further 2026-01-30 to 2026-02-02)
- Predicate name-mismatch validation, and automatic wrapping of raw
  callables (functions, lambdas, `nn.Module`s) into `Predicate` instances
  (2025-12-29, 2025-12-31)
- `ConsistencyChecker` given a new interface with multiclass and
  `nn.Module` predicate support (2026-02-09 to 2026-02-11)
- Boolean operators (`And`, `Or`, `Not`, `Implies`, `Equivalent`)
  re-exported directly from SymPy instead of being reimplemented
  (2026-03-05)

### Added - Documentation and Examples

- GitHub Pages documentation site, including concepts and
  custom-compilers pages (2026-03-04 to 2026-03-05)
- Example notebooks: Semi-Supervised MNIST, MNIST Addition, Symmetry
  Constraints, Triplet Learning, Custom Compilers, plus a notebooks
  README index (2026-02-02 to 2026-06-15)
- Contribution guide and pre-commit hook documentation
  (2026-02-04, 2026-03-02)

### Added - CI and Tooling

- GitHub Actions workflow for tests, mypy, and pylint (2026-02-04)
- Local pre-commit hook covering tests, mypy, and pylint
  (2026-01-18, 2026-03-02)
- Project license added (2026-01-19)

### Fixed

- Arity validation and caching bugs affecting multi-output and custom
  `nn.Module` predicates (2026-01-11 to 2026-02-10)
- Quantifier handling: support for multiple variables, boolean-check
  compatibility with compiled expressions, and SymPy operator
  compatibility via `Boolean` inheritance (2026-02-04 to 2026-02-07)
- Removed the unused `numpy` dependency and a misleading runtime warning
  (2026-03-05)

## [0.1.0] - 2025-12-27

### Added

- Initial project structure: `LogicLoss`/`Predicate` core, t-norm
  implementations (Product, Lukasiewicz, Godel), SymPy-based logic
  expression parsing, PyTorch-differentiable batch evaluation
  (2025-12-23)
- R-Product and S-Product t-norms, replacing `ProductTNorm` as the
  default (2025-12-24)
- `ConsistencyChecker` for hard boolean-logic evaluation on batches
  (2025-12-27)
- Initial `examples/` directory and README (2025-12-23, 2025-12-27)

### Changed

- Refactored t-norms into a dedicated `tnorms/` subpackage (2025-12-27)
