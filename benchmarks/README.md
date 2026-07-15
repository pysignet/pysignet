# Benchmarks

Manual, dev-facing timing scripts. Not part of `pytest tests/` or the
pre-commit hook -- timing assertions are flaky in CI, so these are run by
hand when tuning a performance-sensitive constant or validating a rollout
decision.

## `jit_vs_eager.py`

Compares eager vs. `jit=True` evaluation (TODO.md 2.21) across formula
shapes and sizes. Used to validate `LogicCompiler.JIT_SIZE_THRESHOLD` and
to gather real numbers before flipping `jit`'s default (Phase 2 of the
rollout plan).

```bash
python benchmarks/jit_vs_eager.py
python benchmarks/jit_vs_eager.py --warmup 5 --reps 50   # more stable numbers
```

Prints a results table, a suggested `JIT_SIZE_THRESHOLD` based on where
`jit` actually starts beating eager, and writes a timestamped CSV to
`results/` (gitignored -- these are outputs, not source, regenerate them
whenever you need fresh numbers).
