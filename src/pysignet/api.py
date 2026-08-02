"""Convenience API for logic compilation and evaluation."""

from collections.abc import Callable
from typing import cast

import sympy as sp
import torch

from pysignet.compilation import (
    LinearThresholdUnitCompiler,
    SemanticLossCompiler,
    TNormCompiler,
    case_split,
)
from pysignet.compilation.base import LogicCompiler
from pysignet.compilation.compiled_expression import CompiledExpression
from pysignet.context import EvaluationContext
from pysignet.eval.report import ConsistencyReport
from pysignet.logic import extract_variables
from pysignet.loss import LogicLoss
from pysignet.predicate import Predicate
from pysignet.tnorms import MixedTNorm, TNorm


def compile_logic(
    expr: sp.Basic,
    predicates: dict[str, Predicate | Callable[..., torch.Tensor]],
    mode: str = "tnorm",
    tnorm: TNorm | None = None,
    alpha: float = 1.0,
    max_atoms: int | None = None,
    group_by_evidence: bool = False,
) -> CompiledExpression:
    """Compile logic expression into a CompiledExpression.

    This is the main entry point for most users. It compiles a SymPy logic
    expression into a CompiledExpression that can evaluate satisfaction
    degrees per-batch. Wrap the result in LogicLoss for loss computation
    and batch quantification.

    Args:
        expr: SymPy logic expression (e.g., sp.And(P(X), Q(X)))
        predicates: Dict mapping predicate names to Predicate objects or
            callables that produce torch Tensors
        mode: Compilation mode - 'tnorm' (default), 'ltu', or 'semantic'
        tnorm: T-norm for mode='tnorm' (default: MixedTNorm). Ignored
            for other modes.
        alpha: Sigmoid sharpness for mode='ltu' (default: 1.0). Larger
            values make AND/OR thresholds sharper.
        max_atoms: Maximum unique ground atoms for mode='semantic'
            (default: SemanticLossCompiler.DEFAULT_MAX_ATOMS). Ignored
            for other modes. See SemanticLossCompiler's docstring for
            why this guard exists.
        group_by_evidence: If True, expr must be exactly
            `ForAll(S, domain, Implies(Cond(S), Body(S)))` where Cond
            represents observed per-example hard evidence (e.g. a
            training label), not a soft/learned prediction. Groups
            batch elements by which branch is actually live in a given
            batch and evaluates each group only against its own
            compiled Body(s) -- branches absent from the batch are
            never evaluated. Compiler-agnostic (works with any mode);
            unlike mode='semantic's own case-split handling (which
            evaluates every branch, weighted, to avoid one huge SDD),
            this is a genuine compute reduction, not a circuit-size
            one. v1 scope: expr must match the shape at the top level
            (no nested support yet -- see TODO.md 2.22). Default False.

    Returns:
        CompiledExpression instance for evaluating satisfaction degrees

    Raises:
        ValueError: If unknown mode specified, tnorm= given with
            mode='ltu' or mode='semantic', or group_by_evidence=True
            with an expr that doesn't match the required shape
        ImportError: If mode='semantic' and PySDD is not installed

    Examples:
        Default (MixedTNorm):

        ```python
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.And(P(X), Q(X))
        compiled = compile_logic(expr, {"P": model_p, "Q": model_q})
        satisfaction = compiled(X=x)  # shape: (batch_size,)
        ```

        With a custom t-norm:

        ```python
        from pysignet.tnorms import LukasiewiczTNorm
        compiled = compile_logic(expr, predicates, tnorm=LukasiewiczTNorm())
        ```

        With the LTU compiler:

        ```python
        compiled = compile_logic(expr, predicates, mode='ltu', alpha=2.0)
        ```

        With the semantic loss compiler:

        ```python
        compiled = compile_logic(expr, predicates, mode='semantic')
        ```

        With hard-evidence branch grouping (e.g. a per-example label
        selecting which of several bodies applies):

        ```python
        compiled = compile_logic(expr, predicates, group_by_evidence=True)
        ```
    """
    # Auto-wrap raw callables in Predicate objects
    wrapped_predicates: dict[str, Predicate | Callable[..., torch.Tensor]] = {}
    for key, value in predicates.items():
        if isinstance(value, Predicate):
            # Already a Predicate, use as-is
            wrapped_predicates[key] = value
        elif callable(value):
            # Raw callable (function, lambda, nn.Module) - auto-wrap
            wrapped_predicates[key] = Predicate(value)
        else:
            # Not callable - raise helpful error
            raise TypeError(
                f"Predicate '{key}' must be callable (function, lambda, "
                f"nn.Module) or a Predicate instance, "
                f"got {type(value).__name__}"
            )

    compiler: (
        TNormCompiler | LinearThresholdUnitCompiler | SemanticLossCompiler
    )
    if mode == "tnorm":
        tnorm_instance = tnorm or MixedTNorm()
        compiler = TNormCompiler(tnorm=tnorm_instance)
    elif mode == "ltu":
        if tnorm is not None:
            raise ValueError(
                "tnorm= is only valid with mode='tnorm'. "
                "Use alpha= to configure the LTU compiler."
            )
        compiler = LinearThresholdUnitCompiler(mode="soft", alpha=alpha)
    elif mode == "semantic":
        if tnorm is not None:
            raise ValueError(
                "tnorm= is only valid with mode='tnorm'. "
                "Use max_atoms= to configure the semantic loss compiler."
            )
        compiler = (
            SemanticLossCompiler(max_atoms=max_atoms)
            if max_atoms is not None
            else SemanticLossCompiler()
        )
    else:
        raise NotImplementedError(
            f"Mode '{mode}' is not yet implemented. "
            f"Supported modes: 'tnorm', 'ltu', 'semantic'."
        )

    if group_by_evidence:
        return _compile_with_evidence_grouping(
            compiler, expr, wrapped_predicates
        )

    # Compile the expression with wrapped predicates
    # Returns CompiledExpression (with compiler reference)
    return compiler.compile(expr, wrapped_predicates)


def _compile_with_evidence_grouping(
    compiler: LogicCompiler,
    expr: sp.Basic,
    wrapped_predicates: dict[str, Predicate | Callable[..., torch.Tensor]],
) -> CompiledExpression:
    """Implement group_by_evidence=True (see compile_logic's docstring
    and TODO.md 2.22 for the full design).

    v1 scope: expr must be exactly a qualifying
    ForAll(S, domain, Implies(Cond(S), Body(S))) -- not nested inside a
    larger formula.

    Args:
        compiler: The LogicCompiler instance selected by mode= (used
            both to compile each branch and to evaluate Cond).
        expr: The un-expanded expression passed to compile_logic.
        wrapped_predicates: Already-wrapped Predicate objects.

    Returns:
        A CompiledExpression that groups batch elements by evidence at
        every call.

    Raises:
        ValueError: If expr does not match the required shape at the
            top level.
    """
    candidates = case_split.eligible_candidates(
        expr,
        compiler._expand_quantifiers,  # pylint: disable=protected-access
    )
    matching = [c for c in candidates if c.node is expr]
    if len(matching) != 1:
        raise ValueError(
            "group_by_evidence=True requires expr to be exactly "
            "ForAll(S, domain, Implies(Cond(S), Body(S))) at the top "
            "level (Cond representing observed per-example hard "
            "evidence). "
            + (
                "No such pattern was found."
                if not candidates
                else "A matching pattern was found, but only nested "
                "inside a larger expression -- nested group_by_evidence "
                "is not yet supported (see TODO.md 2.22)."
            )
        )
    candidate = matching[0]
    # By this point every value is a genuine Predicate (compile_logic's
    # auto-wrap loop already ran); narrow the type for the calls below,
    # which require dict[str, Predicate] specifically.
    predicates_wrapped = cast(dict[str, Predicate], wrapped_predicates)

    branch_compiled = {
        value: compiler.compile(
            case_split.substitute_variable(
                candidate.body_template, candidate.variable, value
            ),
            wrapped_predicates,
        )
        for value in candidate.domain
    }

    expanded = compiler._expand_quantifiers(  # pylint: disable=protected-access
        expr
    )
    free_vars = extract_variables(expanded)

    def compiled_logic(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        ctx = EvaluationContext()
        cond_values = [
            compiler._evaluate_predicate_application(  # pylint: disable=protected-access
                case_split.substitute_variable(
                    candidate.cond_template, candidate.variable, value
                ),
                inputs,
                predicates_wrapped,
                ctx,
            )
            for value in candidate.domain
        ]
        case_split.validate_hard_evidence(cond_values)

        stacked = torch.stack(cond_values, dim=0)
        live_index = stacked.argmax(dim=0)
        batch_size = cond_values[0].shape[0]

        # dtype/device come from a branch's own output (Body), not
        # Cond's -- Cond is only used for routing, and the two may
        # legitimately differ (e.g. a hard-evidence label tensor vs. a
        # real nn.Module's float32 output).
        result: torch.Tensor | None = None
        for i, value in enumerate(candidate.domain):
            mask = live_index == i
            if not bool(mask.any()):
                continue
            idx = mask.nonzero(as_tuple=True)[0]
            gathered_inputs = {k: v[idx] for k, v in inputs.items()}
            # mypy cannot verify a **dict call against a signature
            # mixing specific keyword-only params (return_boolean,
            # log_mode) with **variable_bindings: Tensor -- known
            # limitation, not a real type error (gathered_inputs is
            # dict[str, Tensor], matching **variable_bindings exactly).
            branch_result = branch_compiled[value](
                **gathered_inputs  # type: ignore[arg-type]
            )
            if result is None:
                result = torch.zeros(
                    batch_size,
                    dtype=branch_result.dtype,
                    device=branch_result.device,
                )
            result = result.index_copy(0, idx, branch_result)
        if result is None:
            # No example in the batch matched any branch (e.g. an
            # empty batch) -- fall back to Cond's dtype/device so an
            # empty-but-valid tensor is still returned, not a crash.
            sample = cond_values[0]
            result = torch.zeros(
                batch_size, dtype=sample.dtype, device=sample.device
            )
        return result

    return CompiledExpression(
        compiled_logic=compiled_logic,
        free_variables=set(v.name for v in free_vars),
        predicates=predicates_wrapped,
        compiler=compiler,
        expr=expr,
    )


def logic_to_loss(
    expr: sp.Basic,
    predicates: dict[str, Predicate | Callable[..., torch.Tensor]],
    mode: str = "tnorm",
    tnorm: TNorm | None = None,
    alpha: float = 1.0,
    max_atoms: int | None = None,
    group_by_evidence: bool = False,
    post_processing: str | Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> LogicLoss:
    """Compile logic expression and wrap in LogicLoss.

    Convenience function that compiles a logic expression and wraps it
    in a LogicLoss for training. Equivalent to:

        compiled = compile_logic(expr, predicates, mode=mode, tnorm=tnorm,
                                 alpha=alpha, max_atoms=max_atoms,
                                 group_by_evidence=group_by_evidence)
        LogicLoss(compiled, post_processing=post_processing)

    Args:
        expr: SymPy logic expression (e.g., sp.And(P(X), Q(X)))
        predicates: Dict mapping predicate names to Predicate objects or
            callables that produce torch Tensors
        mode: Compilation mode - 'tnorm' (default), 'ltu', or 'semantic'
        tnorm: T-norm for mode='tnorm' (default: MixedTNorm). Ignored
            for other modes.
        alpha: Sigmoid sharpness for mode='ltu' (default: 1.0).
        max_atoms: Maximum unique ground atoms for mode='semantic'.
            Ignored for other modes.
        group_by_evidence: See compile_logic's docstring. Default False.
        post_processing: Post-processing mode - 'log', 'linear', callable,
            or None to use the compiler's recommendation (default)

    Returns:
        LogicLoss instance ready for computing satisfaction and loss

    Examples:
        ```python
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.Implies(P(X), Q(X))
        logic_loss = logic_to_loss(expr, {"P": model_p, "Q": model_q})
        loss = logic_loss.loss(X=x)
        ```

        With LTU compiler:

        ```python
        logic_loss = logic_to_loss(expr, predicates, mode='ltu', alpha=2.0)
        ```

        With semantic loss:

        ```python
        logic_loss = logic_to_loss(expr, predicates, mode='semantic')
        ```
    """
    compiled = compile_logic(
        expr,
        predicates,
        mode=mode,
        tnorm=tnorm,
        alpha=alpha,
        max_atoms=max_atoms,
        group_by_evidence=group_by_evidence,
    )
    return LogicLoss(compiled, post_processing=post_processing)


def consistency_report(
    expression: sp.Basic | dict[str, sp.Basic],
    predicates: dict[str, Predicate | Callable[..., torch.Tensor]],
) -> ConsistencyReport:
    """Create a ConsistencyReport for measuring formula consistency.

    Convenience function that auto-wraps raw callables in Predicate
    objects and creates a ConsistencyReport. Equivalent to:

        ConsistencyReport(expression, predicates)

    Accepts a single SymPy expression or a dict mapping constraint
    names to expressions for multi-constraint reporting.

    The antecedent for conditional violation is auto-detected:
    Implies(A, B) uses A; any other formula uses sp.true.

    Args:
        expression: SymPy logic expression or dict of named
            expressions (e.g., {"sym": expr1, "trans": expr2}).
        predicates: Dict mapping predicate names to Predicate objects or
            callables that produce torch Tensors

    Returns:
        ConsistencyReport instance for accumulating and querying metrics

    Example:
        ```python
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.Implies(P(X), Q(X))
        report = consistency_report(expr, {"P": model_p, "Q": model_q})
        for x_batch in dataloader:
            report.eval(X=x_batch)
        print(report.global_violation())
        ```
    """
    wrapped_predicates: dict[str, Predicate] = {}
    for key, value in predicates.items():
        if isinstance(value, Predicate):
            wrapped_predicates[key] = value
        elif callable(value):
            wrapped_predicates[key] = Predicate(value)
        else:
            raise TypeError(
                f"Predicate '{key}' must be callable (function, lambda, "
                f"nn.Module) or a Predicate instance, "
                f"got {type(value).__name__}"
            )
    return ConsistencyReport(expression, wrapped_predicates)
