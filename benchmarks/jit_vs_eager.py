"""Benchmark: eager vs. jit=True evaluation (TODO.md 2.21).

Standalone script, not part of the pytest suite -- timing assertions are
flaky in CI, so this is a manually-run dev tool for validating the
JIT_SIZE_THRESHOLD heuristic and the Phase 1 -> Phase 2 rollout decision
(see TODO.md 2.21 and SEMANTIC_LOSS_DESIGN.md's cross-reference note).

Usage:
    python benchmarks/jit_vs_eager.py [--warmup N] [--reps N] [--seed N]
    python benchmarks/jit_vs_eager.py --batch-sizes 32 4096 65536

Measures, per (compiler, formula shape, leaf count, batch size, jit)
config:
- median_ms: steady-state per-call latency (post-warmup, N measured reps)
- first_call_ms: latency of the very first call, where torch.compile's
  Dynamo tracing/Inductor compilation happens for jit=True configs
- speedup: eager median_ms / jit median_ms, for matching configs

Formula shapes:
- "and": pure conjunction, swept across leaf counts straddling
  JIT_SIZE_THRESHOLD, to isolate the size effect and empirically check
  the threshold's value.
- "mixed": a single 32-leaf formula combining And/Or/Not/Implies/
  Equivalent, to confirm the sweep's pure-And numbers generalize.
- "addition_scale": the actual MNIST Addition constraint shape
  (ForAll/Implies/Exists over synthetic constant predicates, not real
  MNIST data) -- the ~190-leaf scale that TODO.md 2.21's Step 1 tests
  explicitly did not cover.

Batch size matters as much as leaf count (discovered while digging into
why LinearThresholdUnitCompiler showed no jit benefit -- see TODO.md
2.21): torch.compile has a fixed per-call dispatch/guard-check overhead
(tens of microseconds, independent of workload), so jit only pays off
once actual per-call compute exceeds that floor. At small batch sizes,
eager computation on these tiny per-node tensor ops is *already* below
that floor, and jit can only add overhead.

Results are written to benchmarks/results/ (gitignored -- this is
output, not source) as a timestamped CSV, and also printed as a table.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from pysignet import (
    And,
    Equivalent,
    Implies,
    Not,
    Or,
    Predicate,
    Symbol,
    TNormCompiler,
    Variable,
)
from pysignet.compilation.base import LogicCompiler
from pysignet.compilation.ltu_compiler import LinearThresholdUnitCompiler
from pysignet.logic.quantifier import Exists, ForAll

AND_SWEEP_SIZES = [2, 4, 8, 16, 32, 64, 128]
DEFAULT_BATCH_SIZES = [32, 512, 4096, 32768]

# (expr, predicates, bindings_fn(batch_size) -> dict[str, Tensor])
Shape = tuple[Any, dict[str, Predicate], Callable[[int], dict[str, torch.Tensor]]]


def _make_constant_predicate(value: float):
    def predicate(x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[0]) * value

    return predicate


def _constant_predicates(symbols: Any, values: list[float]) -> dict[str, Predicate]:
    return {
        str(sym): Predicate(_make_constant_predicate(value))
        for sym, value in zip(symbols, values, strict=True)
    }


def build_and_formula(n: int) -> Shape:
    """Pure conjunction over n distinct atoms."""
    x_var = Variable("X")
    names = " ".join(f"P{i}" for i in range(n))
    symbols = Symbol(names) if n > 1 else (Symbol(names),)
    apps = [symbols[i](x_var) for i in range(n)]
    expr = apps[0]
    for app in apps[1:]:
        expr = expr & app

    predicates = _constant_predicates(
        symbols, [0.05 * (i % 15 + 1) for i in range(n)]
    )

    def bindings_fn(batch_size: int) -> dict[str, torch.Tensor]:
        return {x_var.name: torch.randn(batch_size, 4)}

    return expr, predicates, bindings_fn


def build_mixed_formula() -> Shape:
    """32-leaf formula mixing And/Or/Not/Implies/Equivalent."""
    n = 32
    x_var = Variable("X")
    names = " ".join(f"M{i}" for i in range(n))
    symbols = Symbol(names)
    apps = [symbols[i](x_var) for i in range(n)]

    or_part = Or(*apps[0:8])
    not_part = And(*[Not(a) for a in apps[8:16]])
    implies_parts = [
        Implies(apps[i], apps[i + 1]) for i in range(16, 24, 2)
    ]
    equiv_parts = [
        Equivalent(apps[i], apps[i + 1]) for i in range(24, 32, 2)
    ]
    expr = And(or_part, not_part, *implies_parts, *equiv_parts)

    predicates = _constant_predicates(
        symbols, [0.05 * (i % 15 + 1) for i in range(n)]
    )

    def bindings_fn(batch_size: int) -> dict[str, torch.Tensor]:
        return {x_var.name: torch.randn(batch_size, 4)}

    return expr, predicates, bindings_fn


def build_addition_scale_formula() -> Shape:
    """The real MNIST Addition constraint shape, with synthetic constant
    predicates instead of real MNIST models -- same quantifier expansion
    (up to 19 * 10 = 190 leaf Digit(...) applications), no torchvision
    dependency."""
    sum_sym = Symbol("Sum")
    digit_sym = Symbol("Digit")
    x1, x2, s_actual, s_var, i_var = Variable("X1 X2 S_actual S I")

    expr = ForAll(
        s_var,
        range(19),
        Implies(
            sum_sym(s_actual, s_var),
            Exists(
                i_var,
                range(10),
                And(digit_sym(x1, i_var), digit_sym(x2, s_var - i_var)),
            ),
        ),
    )

    def digit_pred(x: torch.Tensor, digit_idx: int) -> torch.Tensor:
        if not 0 <= digit_idx <= 9:
            return torch.zeros(x.shape[0])
        return torch.ones(x.shape[0]) * (0.05 * (digit_idx + 1))

    predicates: dict[str, Predicate] = {
        "Sum": Predicate(
            lambda s_actual_t, s_t: (s_actual_t == s_t).float()
        ),
        "Digit": Predicate(digit_pred),
    }

    def bindings_fn(batch_size: int) -> dict[str, torch.Tensor]:
        return {
            "X1": torch.randn(batch_size, 4),
            "X2": torch.randn(batch_size, 4),
            "S_actual": torch.full((batch_size,), 9),
        }

    return expr, predicates, bindings_fn


def actual_leaf_count(compiler: LogicCompiler, expr: Any) -> int:
    """Number of unique leaf atoms after quantifier expansion (private
    introspection -- fine for a dev-only benchmark script)."""
    expanded = compiler._expand_quantifiers(expr)  # noqa: SLF001
    return len(compiler._collect_leaves(expanded))  # noqa: SLF001


def measure_one(
    compiler: LogicCompiler,
    expr: Any,
    predicates: dict[str, Predicate],
    bindings: dict[str, torch.Tensor],
    warmup: int,
    reps: int,
) -> tuple[float, float]:
    """Returns (median_ms, first_call_ms) for one compiled expression."""
    compiled = compiler.compile(expr, predicates)

    with torch.no_grad():
        t0 = time.perf_counter()
        compiled(**bindings)
        first_call_ms = (time.perf_counter() - t0) * 1000

        for _ in range(warmup):
            compiled(**bindings)

        samples = []
        for _ in range(reps):
            t0 = time.perf_counter()
            compiled(**bindings)
            samples.append((time.perf_counter() - t0) * 1000)

    return statistics.median(samples), first_call_ms


class _ForceJitThreshold:
    """Temporarily override LogicCompiler.JIT_SIZE_THRESHOLD to 1, so
    every jit=True config in the sweep genuinely engages torch.compile
    regardless of its nominal size. Without this, sizes below the
    *current* threshold never reach torch.compile at all when jit=True
    -- their "speedup" would just be eager-vs-eager noise, making the
    crossover analysis meaningless. This is what lets the sweep answer
    "where does jit actually start winning", independent of whatever
    JIT_SIZE_THRESHOLD happens to be set to right now.
    """

    def __enter__(self) -> None:
        self._original = LogicCompiler.JIT_SIZE_THRESHOLD
        LogicCompiler.JIT_SIZE_THRESHOLD = 1

    def __exit__(self, *exc_info: object) -> None:
        LogicCompiler.JIT_SIZE_THRESHOLD = self._original


def run_all(
    warmup: int, reps: int, batch_sizes: list[int]
) -> list[dict[str, Any]]:
    compiler_factories = {
        "TNormCompiler": lambda jit: TNormCompiler(jit=jit),
        "LinearThresholdUnitCompiler": (
            lambda jit: LinearThresholdUnitCompiler(jit=jit)
        ),
    }

    shapes: list[tuple[str, Shape]] = [
        ("and", build_and_formula(n)) for n in AND_SWEEP_SIZES
    ]
    shapes.append(("mixed", build_mixed_formula()))
    shapes.append(("addition_scale", build_addition_scale_formula()))

    rows: list[dict[str, Any]] = []
    total = (
        len(shapes) * len(compiler_factories) * len(batch_sizes) * 2
    )
    done = 0

    for shape_name, (expr, predicates, bindings_fn) in shapes:
        for compiler_name, factory in compiler_factories.items():
            n_leaves = actual_leaf_count(factory(False), expr)
            for batch_size in batch_sizes:
                bindings = bindings_fn(batch_size)
                for jit in (False, True):
                    done += 1
                    print(
                        f"[{done}/{total}] {compiler_name} "
                        f"shape={shape_name} n_leaves={n_leaves} "
                        f"batch={batch_size} jit={jit} ...",
                        end=" ",
                        flush=True,
                    )
                    compiler = factory(jit)
                    with _ForceJitThreshold():
                        median_ms, first_call_ms = measure_one(
                            compiler, expr, predicates, bindings,
                            warmup, reps,
                        )
                    print(
                        f"median={median_ms:.3f}ms "
                        f"first_call={first_call_ms:.3f}ms"
                    )
                    rows.append(
                        {
                            "compiler": compiler_name,
                            "shape": shape_name,
                            "n_leaves": n_leaves,
                            "batch_size": batch_size,
                            "jit": jit,
                            "median_ms": median_ms,
                            "first_call_ms": first_call_ms,
                        }
                    )

    _add_speedup(rows)
    return rows


def _add_speedup(rows: list[dict[str, Any]]) -> None:
    """Fill in speedup = eager_median / jit_median for matching configs."""
    eager_by_key = {
        (r["compiler"], r["shape"], r["n_leaves"], r["batch_size"]): (
            r["median_ms"]
        )
        for r in rows
        if not r["jit"]
    }
    for row in rows:
        key = (
            row["compiler"], row["shape"], row["n_leaves"],
            row["batch_size"],
        )
        eager_ms = eager_by_key.get(key)
        row["speedup"] = (
            eager_ms / row["median_ms"]
            if row["jit"] and eager_ms
            else None
        )


def print_table(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'compiler':<28} {'shape':<15} {'n_leaves':>8} {'batch':>7} "
        f"{'jit':>5} {'median_ms':>10} {'first_call_ms':>14} "
        f"{'speedup':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        speedup = f"{row['speedup']:.2f}x" if row["speedup"] else "--"
        print(
            f"{row['compiler']:<28} {row['shape']:<15} "
            f"{row['n_leaves']:>8} {row['batch_size']:>7} "
            f"{str(row['jit']):>5} {row['median_ms']:>10.3f} "
            f"{row['first_call_ms']:>14.3f} {speedup:>8}"
        )


def print_suggested_threshold(rows: list[dict[str, Any]]) -> None:
    """Per (compiler, batch size), find the smallest 'and' sweep leaf
    count at or above which jit speedup stays > 1.0x for every larger
    leaf count too (filters single-point noise -- a lone size that
    happens to edge past 1.0x while neighbors don't is not a real
    crossover). Batch size is swept because it governs whether jit pays
    off at all, at least as much as leaf count does (see module
    docstring)."""
    print()
    print(f"Current JIT_SIZE_THRESHOLD = {LogicCompiler.JIT_SIZE_THRESHOLD}")

    by_key: dict[tuple[str, int], list[tuple[int, float]]] = {}
    for row in rows:
        if row["shape"] != "and" or row["speedup"] is None:
            continue
        key = (row["compiler"], row["batch_size"])
        by_key.setdefault(key, []).append(
            (row["n_leaves"], row["speedup"])
        )

    for (compiler_name, batch_size), points in sorted(by_key.items()):
        points.sort()
        crossover = None
        for i, (n, _) in enumerate(points):
            if all(s > 1.0 for _, s in points[i:]):
                crossover = n
                break
        label = f"{compiler_name} @ batch={batch_size}"
        if crossover is None:
            print(f"{label}: jit never consistently beat eager.")
        else:
            print(f"{label}: consistent speedup from {crossover} leaves onward.")


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "compiler",
                "shape",
                "n_leaves",
                "batch_size",
                "jit",
                "median_ms",
                "first_call_ms",
                "speedup",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    rows = run_all(args.warmup, args.reps, args.batch_sizes)

    print()
    print_table(rows)
    print_suggested_threshold(rows)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = (
        Path(__file__).parent / "results" / f"jit_vs_eager_{timestamp}.csv"
    )
    write_csv(rows, output_path)


if __name__ == "__main__":
    main()
