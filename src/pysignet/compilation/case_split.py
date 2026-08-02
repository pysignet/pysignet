"""Case-split detection and substitution: compiler-agnostic.

Finds ForAll(S, domain, Implies(Cond(S), Body(S))) subtrees -- anywhere
in the expression, not just at the top level -- whose atoms are
disjoint from every atom used elsewhere in the containing formula. Pure
SymPy-level logic, with no dependency on any particular compiler, used
by two independent features:

- `SemanticLossCompiler` (SEMANTIC_LOSS_DESIGN.md Section 10): replaces
  each eligible subtree with a synthetic ground atom so the (now
  smaller) "outer" expression compiles through the ordinary,
  unmodified semantic loss pipeline. Motivated by SDD circuit-size
  blowup.
- `group_by_evidence=` on `compile_logic`/`logic_to_loss` (TODO.md
  2.22): groups batch elements by which branch is actually live and
  evaluates each group only against its own compiled Body(s), for
  TNormCompiler/LinearThresholdUnitCompiler. Motivated by wasted
  computation, not circuit size -- no SDD involved.

This module only handles compile-time-decidable structure (shape
matching and atom disjointness) plus the runtime hard-evidence check
both features need (`validate_hard_evidence`). Everything else
(substitution strategy, per-branch compilation, evaluation) is
feature-specific and lives in semantic_compiler.py or api.py
respectively.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import sympy as sp
import torch

from pysignet.compilation.base import collect_leaves
from pysignet.logic.expansion import (
    _substitute_variable,  # pylint: disable=protected-access
)
from pysignet.logic.quantifier import ForAll, Quantifier
from pysignet.logic.variable import VariableSymbol
from pysignet.symbols import PredicateApplication

_HARD_EVIDENCE_TOLERANCE = 1e-4


def validate_hard_evidence(cond_probs: list[torch.Tensor]) -> None:
    """Verify Cond's values are hard 0/1 with exactly one true per
    batch element.

    Shared by both case-split features (SemanticLossCompiler's and
    group_by_evidence's): each relies on Cond representing observed
    per-example evidence, not a soft/uncertain prediction, for its
    respective transformation to be exact -- see
    SEMANTIC_LOSS_DESIGN.md Section 10 for the derivation and the
    numerical counter-example that motivated this check.

    Args:
        cond_probs: One (batch_size,) tensor per domain value.

    Raises:
        ValueError: If any value isn't close to 0 or 1, or if the
            per-example sum across domain values isn't close to 1.
    """
    stacked = torch.stack(cond_probs, dim=0)
    is_hard = bool(
        (
            (stacked < _HARD_EVIDENCE_TOLERANCE)
            | (stacked > 1.0 - _HARD_EVIDENCE_TOLERANCE)
        ).all()
    )
    sums = stacked.sum(dim=0)
    sums_ok = bool(
        torch.allclose(
            sums, torch.ones_like(sums), atol=_HARD_EVIDENCE_TOLERANCE
        )
    )
    if not (is_hard and sums_ok):
        raise ValueError(
            "Case-split condition must be hard evidence (0/1 values, "
            "exactly one true per example) -- this decomposition is "
            "only exact for observed evidence, not soft/uncertain "
            "predictions (see SEMANTIC_LOSS_DESIGN.md Section 10). "
            f"Got per-example sums: {sums.tolist()}, all values hard: "
            f"{is_hard}."
        )


@dataclass(frozen=True)
class CaseSplitCandidate:
    """A ForAll(S, domain, Implies(Cond(S), Body(S))) subtree found in
    an expression, before eligibility (atom-disjointness) is checked.

    Attributes:
        node: The original ForAll node, used for identity-based
            substitution (never equality-based: two structurally equal
            but distinct ForAll occurrences must not be conflated).
        variable: The quantified variable S.
        domain: Materialized domain values (list, not a lazy iterable).
        cond_template: Cond(..., S) -- still contains the variable S,
            substituted per branch during eligibility checking and
            compilation.
        body_template: Body(S) -- likewise still contains S.
    """

    node: ForAll
    variable: VariableSymbol
    domain: list[Any]
    cond_template: PredicateApplication
    body_template: sp.Basic


def substitute_variable(
    expr: sp.Basic, variable: VariableSymbol, value: Any
) -> sp.Basic:
    """Substitute a variable with a value, including inside
    PredicateApplication.application_args (which ordinary SymPy
    substitution cannot reach, since PredicateApplication.args is
    hardcoded empty). Thin wrapper around the existing quantifier
    expansion internals.

    Args:
        expr: Expression to substitute within.
        variable: Variable to replace.
        value: Value to replace it with.

    Returns:
        Expression with the variable replaced.
    """
    return _substitute_variable(expr, variable, value)


def find_case_split_candidates(expr: sp.Basic) -> list[CaseSplitCandidate]:
    """Recursively find maximal qualifying ForAll subtrees anywhere in
    expr (not just at the top level).

    Does not recurse into a matched candidate's own Body(S) looking for
    further nested candidates (v1 scope: bounded recursion depth of 1).

    Args:
        expr: Un-expanded expression to search.

    Returns:
        List of candidates in the order found (pre-order traversal).
    """
    candidates: list[CaseSplitCandidate] = []

    def walk(node: sp.Basic) -> None:
        if isinstance(node, ForAll) and _is_case_split_shape(node):
            candidates.append(_build_candidate(node))
            return
        if isinstance(node, Quantifier):
            walk(node.body)
            return
        for child in getattr(node, "args", ()):
            walk(child)

    walk(expr)
    return candidates


def _is_case_split_shape(node: ForAll) -> bool:
    """Check node matches ForAll(S, domain, Implies(Cond(S), Body(S)))."""
    if isinstance(node.variable, list):
        return False
    if not isinstance(node.body, sp.Implies):
        return False
    antecedent = node.body.args[0]
    if not isinstance(antecedent, PredicateApplication):
        return False
    occurrences = sum(
        1 for arg in antecedent.application_args if arg == node.variable
    )
    return occurrences == 1


def _build_candidate(node: ForAll) -> CaseSplitCandidate:
    antecedent, consequent = node.body.args
    assert isinstance(antecedent, PredicateApplication)
    return CaseSplitCandidate(
        node=node,
        variable=node.variable,
        domain=list(node.domain),
        cond_template=antecedent,
        body_template=consequent,
    )


def replace_node(
    expr: sp.Basic, target: sp.Basic, replacement: sp.Basic
) -> sp.Basic:
    """Rebuild expr with the object identical to target replaced.

    Matches by identity (`is`), not equality: two structurally equal
    but distinct occurrences of a subtree must not be conflated.
    Reconstructs Quantifier nodes explicitly (preserving domain, which
    Quantifier.args excludes) rather than via generic
    node.func(*node.args), which would drop it.

    Args:
        expr: Expression to rebuild.
        target: The specific node object to replace.
        replacement: Its replacement.

    Returns:
        A new expression with target replaced, sharing structure with
        expr wherever nothing changed.
    """

    def rebuild(node: sp.Basic) -> sp.Basic:
        if node is target:
            return replacement
        if isinstance(node, Quantifier):
            new_body = rebuild(node.body)
            if new_body is node.body:
                return node
            return type(node)(node.variable, node.domain, new_body)
        args = getattr(node, "args", ())
        if not args:
            return node
        new_args = [rebuild(arg) for arg in args]
        if all(a is b for a, b in zip(new_args, args, strict=True)):
            return node
        return node.func(*new_args)

    return rebuild(expr)


def candidate_atoms(
    candidate: CaseSplitCandidate,
    expand_fn: Callable[[sp.Basic], sp.Basic],
) -> set[PredicateApplication]:
    """All ground atoms referenced anywhere inside a candidate's own
    fully-expanded content: Cond(s) and Body(s) for every s in domain.

    Args:
        candidate: The candidate to collect atoms for.
        expand_fn: Quantifier-expansion function (e.g. a compiler's
            bound _expand_quantifiers), applied to each Body(s) to
            expand any further-nested quantifiers within it.

    Returns:
        Set of unique PredicateApplication atoms.
    """
    atoms: set[PredicateApplication] = set()
    for value in candidate.domain:
        cond_s = substitute_variable(
            candidate.cond_template, candidate.variable, value
        )
        assert isinstance(cond_s, PredicateApplication)
        atoms.add(cond_s)
        body_s = substitute_variable(
            candidate.body_template, candidate.variable, value
        )
        atoms.update(collect_leaves(expand_fn(body_s)))
    return atoms


def outside_atoms(
    expr: sp.Basic,
    candidate: CaseSplitCandidate,
    expand_fn: Callable[[sp.Basic], sp.Basic],
) -> set[PredicateApplication]:
    """All ground atoms referenced anywhere in expr except inside this
    candidate's own subtree (other candidates' atoms ARE included, so
    checking disjointness against this set also catches overlap between
    two different case-split candidates).

    Args:
        expr: The original (un-expanded) expression.
        candidate: The candidate whose own subtree is excluded.
        expand_fn: Quantifier-expansion function.

    Returns:
        Set of unique PredicateApplication atoms used outside candidate.
    """
    replaced = replace_node(expr, candidate.node, sp.true)
    return set(collect_leaves(expand_fn(replaced)))


def eligible_candidates(
    expr: sp.Basic,
    expand_fn: Callable[[sp.Basic], sp.Basic],
) -> list[CaseSplitCandidate]:
    """Find all candidates whose atoms are disjoint from the rest of
    expr -- the only compile-time-decidable eligibility requirement.

    Args:
        expr: Un-expanded expression to search.
        expand_fn: Quantifier-expansion function.

    Returns:
        Eligible candidates, in the order found.
    """
    eligible: list[CaseSplitCandidate] = []
    for candidate in find_case_split_candidates(expr):
        atoms_i = candidate_atoms(candidate, expand_fn)
        atoms_rest = outside_atoms(expr, candidate, expand_fn)
        if atoms_i.isdisjoint(atoms_rest):
            eligible.append(candidate)
    return eligible
