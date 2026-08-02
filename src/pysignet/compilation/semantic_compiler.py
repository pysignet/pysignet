"""Semantic loss compilation strategy using weighted model counting.

Unlike TNormCompiler/LinearThresholdUnitCompiler, which combine truth
degrees locally at each And/Or/Not node, SemanticLossCompiler computes
satisfaction as weighted model counting (WMC) over a compiled Sentential
Decision Diagram (SDD). This makes satisfaction depend only on the
formula's meaning (its set of satisfying assignments), not its syntax --
see SEMANTIC_LOSS_DESIGN.md for the full derivation and references.

Requires the optional PySDD dependency: pip install pysignet[semantic].
"""

import functools
import operator
from collections.abc import Callable
from typing import Any, NamedTuple

import sympy as sp
import torch

from pysignet.compilation.base import LogicCompiler
from pysignet.compilation.compiled_expression import CompiledExpression
from pysignet.context import EvaluationContext
from pysignet.logic import extract_variables
from pysignet.predicate import Predicate
from pysignet.symbols import PredicateApplication

_INSTALL_HINT = (
    "PySDD is required for mode='semantic' (SemanticLossCompiler). "
    "Install it with: pip install pysignet[semantic]"
)


def _import_pysdd() -> Any:
    """Import PySDD's SddManager/Vtree, with a clear install-hint error.

    Returns:
        The pysdd.sdd module.

    Raises:
        ImportError: If PySDD is not installed.
    """
    try:
        import pysdd.sdd  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return pysdd.sdd


class SemanticLossCompiler(LogicCompiler):
    """Compiles logic expressions into semantic loss via a compiled SDD.

    Builds a Sentential Decision Diagram (SDD) once, at compile() time,
    from the expanded SymPy expression, then linearizes it into a flat,
    topologically ordered program (also once, at compile() time --
    mirroring pypsdd's own weighted_model_count()/generate_tf_ac(),
    which the reference implementation's own comment notes is "faster
    than recursive traversal"). Every training-batch call evaluates
    that flat program iteratively: no Python recursion, no per-call
    memo dict, and no further PySDD API calls. Gradients flow to
    predicate parameters through ordinary autograd, since the WMC walk
    is built entirely from +/* tensor operations.

    Unlike TNormCompiler/LinearThresholdUnitCompiler, conjunction() and
    disjunction() on this class are NOT used to evaluate the formula
    itself (WMC is a global, structural computation, not a local
    pointwise one -- see SEMANTIC_LOSS_DESIGN.md Section 3). They exist
    only to satisfy the LogicCompiler ABC and are used exclusively for
    i.i.d. batch reduction (forall/exists quantification over
    independent batch elements), where a plain product t-norm is exactly
    correct.

    Args:
        max_atoms: Maximum number of unique ground atoms (SDD variables)
            allowed in a compiled expression. This is a coarse,
            atom-count-based guard, not a true circuit-size bound: the
            Step 0 feasibility spike (see TODO.md 2.8) found a
            39-atom/190-leaf-conjunct formula (MNIST Addition) already
            compiles to an 840,000-node SDD, while a 10-atom formula
            (exactly_one) compiles to 34 nodes -- circuit size depends
            on formula structure, not atom count alone. Default is
            deliberately conservative; raise it only after checking the
            resulting circuit's size yourself (see SEMANTIC_LOSS_DESIGN.md
            Section 6). Default: 20.

    Example:
        ```python
        compiler = SemanticLossCompiler()
        compiled = compiler.compile(expr, predicates)
        satisfaction = compiled(X=x)  # Returns tensor in [0, 1]
        ```
    """

    DEFAULT_MAX_ATOMS = 20

    def __init__(self, max_atoms: int = DEFAULT_MAX_ATOMS) -> None:
        """Initialize SemanticLossCompiler.

        Args:
            max_atoms: Maximum number of unique ground atoms allowed.
                See class docstring for why this is a coarse guard.
        """
        self.max_atoms = max_atoms

    @property
    def recommended_postprocessing(self) -> str:
        """Semantic loss is literally -log(WMC), so recommend 'log'."""
        return "log"

    def conjunction(self, values: torch.Tensor) -> torch.Tensor:
        """Batch reduction ONLY: product of independent WMC values.

        Not used to evaluate the formula itself -- see class docstring.

        Args:
            values: Tensor of shape (n, ...) with values in [0, 1].

        Returns:
            Tensor of shape (...): product along dim=0.
        """
        return values.prod(dim=0)

    def disjunction(self, values: torch.Tensor) -> torch.Tensor:
        """Batch reduction ONLY: probabilistic sum of independent values.

        Not used to evaluate the formula itself -- see class docstring.

        Args:
            values: Tensor of shape (n, ...) with values in [0, 1].

        Returns:
            Tensor of shape (...): 1 - prod(1 - values) along dim=0.
        """
        return 1.0 - (1.0 - values).prod(dim=0)

    def compile(
        self,
        expr: sp.Basic,
        predicates: dict[str, Predicate | Callable[..., torch.Tensor]],
    ) -> CompiledExpression:
        """Compile a logic expression into a CompiledExpression.

        Args:
            expr: SymPy logic expression
            predicates: Dict mapping predicate names to Predicate
                objects or callables

        Returns:
            CompiledExpression that evaluates satisfaction as WMC over
            a compiled SDD, built once and cached in this closure.

        Raises:
            ValueError: If symbols in expr have no corresponding
                predicates, or if the expression has more unique ground
                atoms than max_atoms.
            ImportError: If PySDD is not installed.
            TypeError: If predicate values are not callable.
        """
        wrapped_predicates = self._wrap_and_validate_predicates(
            expr, predicates
        )
        expanded_expr = self._expand_quantifiers(expr)
        free_vars = extract_variables(expanded_expr)

        leaf_order = self._collect_leaves(expanded_expr)
        if len(leaf_order) > self.max_atoms:
            raise ValueError(
                f"Expression has {len(leaf_order)} unique ground atoms, "
                f"exceeding max_atoms={self.max_atoms}. SDD compilation "
                f"time/size can grow steeply with formula structure "
                f"(see SEMANTIC_LOSS_DESIGN.md Section 6). Restructure "
                f"the expression, or pass a larger max_atoms= only "
                f"after confirming the resulting circuit size is "
                f"acceptable."
            )

        pysdd_sdd = _import_pysdd()
        atom_to_var = {atom: i + 1 for i, atom in enumerate(leaf_order)}
        mgr = _make_sdd_manager(pysdd_sdd, len(leaf_order))
        compiled_sdd = _expr_to_sdd(expanded_expr, mgr, atom_to_var)
        program = _compile_wmc_program(compiled_sdd)

        def compiled_logic(
            inputs: dict[str, torch.Tensor]
        ) -> torch.Tensor:
            ctx = EvaluationContext()
            literal_weights = {
                var: self._evaluate_predicate_application(
                    atom, inputs, wrapped_predicates, ctx
                )
                for atom, var in atom_to_var.items()
            }
            batch_shape, dtype, device = _infer_batch_shape(
                literal_weights, inputs
            )
            return _wmc(
                program, literal_weights, batch_shape, dtype, device
            )

        return CompiledExpression(
            compiled_logic=compiled_logic,
            free_variables=set(v.name for v in free_vars),
            predicates=wrapped_predicates,
            compiler=self,
            expr=expr,
        )


def _make_sdd_manager(pysdd_sdd: Any, num_atoms: int) -> Any:
    """Create an SddManager with exactly num_atoms declared variables.

    Args:
        pysdd_sdd: The pysdd.sdd module (from _import_pysdd()).
        num_atoms: Number of unique ground atoms in the expression.

    Returns:
        An SddManager with num_atoms variables (or 1 unused variable
        if num_atoms is 0, e.g. for an all-constant expression).
    """
    vtree = pysdd_sdd.Vtree(var_count=1)
    mgr = pysdd_sdd.SddManager.from_vtree(vtree)
    for _ in range(max(0, num_atoms - 1)):
        mgr.add_var_after_last()
    return mgr


def _expr_to_sdd(
    node: sp.Basic,
    mgr: Any,
    atom_to_var: dict[PredicateApplication, int],
) -> Any:
    """Recursively compile a SymPy expression into an SDD.

    Mirrors Pylon's SddVisitor, walking a SymPy AST instead of a Python
    AST (see SEMANTIC_LOSS_DESIGN.md Section 5.3). Built once per
    compiled expression; not called again per training batch.

    Args:
        node: SymPy expression node (after quantifier expansion).
        mgr: SddManager to build literals/combinators from.
        atom_to_var: Dict mapping each unique ground PredicateApplication
            to its 1-based SDD variable index.

    Returns:
        The compiled Sdd node for this subexpression.

    Raises:
        ValueError: If node is a bare (nullary) sp.Symbol.
        NotImplementedError: If node's type is not one of the supported
            logical connectives.
    """
    if isinstance(node, PredicateApplication):
        return mgr.literal(atom_to_var[node])

    if isinstance(node, sp.Symbol):
        raise ValueError(
            f"Bare symbol '{node}' is not supported. All predicates "
            f"must be called with at least one variable argument "
            f"(e.g., use P(X) instead of P)."
        )

    if node == sp.true:
        return mgr.true()
    if node == sp.false:
        return mgr.false()

    if isinstance(node, sp.And):
        return functools.reduce(
            operator.and_,
            (_expr_to_sdd(a, mgr, atom_to_var) for a in node.args),
        )
    if isinstance(node, sp.Or):
        return functools.reduce(
            operator.or_,
            (_expr_to_sdd(a, mgr, atom_to_var) for a in node.args),
        )
    if isinstance(node, sp.Not):
        return ~_expr_to_sdd(node.args[0], mgr, atom_to_var)
    if isinstance(node, sp.Implies):
        left, right = node.args
        left_sdd = _expr_to_sdd(left, mgr, atom_to_var)
        right_sdd = _expr_to_sdd(right, mgr, atom_to_var)
        return (~left_sdd) | right_sdd
    if isinstance(node, sp.Equivalent):
        left, right = node.args
        left_sdd = _expr_to_sdd(left, mgr, atom_to_var)
        right_sdd = _expr_to_sdd(right, mgr, atom_to_var)
        return left_sdd.equiv(right_sdd)

    raise NotImplementedError(
        f"Unsupported node type for semantic loss: {type(node)}"
    )


def _infer_batch_shape(
    literal_weights: dict[int, torch.Tensor],
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Size, torch.dtype, torch.device]:
    """Determine shape/dtype/device for true/false leaves.

    Args:
        literal_weights: Dict mapping SDD variable index to its
            (batch_size,) probability tensor. May be empty if the
            expression has no ground atoms (e.g. a bare sp.true).
        inputs: The raw variable bindings passed to compiled_logic,
            used as a fallback when literal_weights is empty.

    Returns:
        Tuple of (batch_shape, dtype, device).

    Raises:
        ValueError: If literal_weights is empty and inputs is also
            empty (no way to determine batch size).
    """
    if literal_weights:
        sample = next(iter(literal_weights.values()))
        return sample.shape, sample.dtype, sample.device

    if not inputs:
        raise ValueError(
            "Inputs dict cannot be empty when the compiled expression "
            "has no predicate atoms."
        )
    sample_input = next(iter(inputs.values()))
    return (
        torch.Size((sample_input.shape[0],)),
        sample_input.dtype,
        sample_input.device,
    )


class _SddOp(NamedTuple):
    """One linearized SDD instruction, referencing children by position.

    Attributes:
        kind: One of "false", "true", "literal", "decision".
        literal: The signed SDD literal (meaningful only for "literal").
        children: (prime_position, sub_position) pairs, each an index
            into the same program list this op belongs to (meaningful
            only for "decision"). Positions always refer to earlier
            entries, since the program is post-order (children before
            parents).
    """

    kind: str
    literal: int
    children: tuple[tuple[int, int], ...]


def _compile_wmc_program(root: Any) -> list[_SddOp]:
    """Flatten a compiled SDD into a linear, position-indexed program.

    Built once per compiled expression, at compile() time. Mirrors
    pypsdd's own weighted_model_count()/generate_tf_ac() (see
    art-ai/pypsdd, sdd.py), which linearize the SDD's DAG once via a
    cached topological sort and then iterate it flatly -- their own
    comment: "faster than recursive traversal of an SDD." This goes a
    step further: every child reference is resolved to an integer
    position at compile time, so evaluating the program (_wmc()) never
    calls back into the PySDD API at all, only +/* tensor arithmetic.

    Args:
        root: The compiled Sdd node (from _expr_to_sdd()).

    Returns:
        A list of _SddOp in post-order (the root is always last).

    Raises:
        ValueError: If a node is not a recognized SDD node type.
    """
    order: list[Any] = []
    positions: dict[int, int] = {}

    def visit(node: Any) -> None:
        if node.id in positions:
            return
        if node.is_decision():
            for prime, sub in node.elements():
                visit(prime)
                visit(sub)
        positions[node.id] = len(order)
        order.append(node)

    visit(root)

    program: list[_SddOp] = []
    for node in order:
        if node.is_false():
            program.append(_SddOp("false", 0, ()))
        elif node.is_true():
            program.append(_SddOp("true", 0, ()))
        elif node.is_literal():
            program.append(_SddOp("literal", node.literal, ()))
        elif node.is_decision():
            children = tuple(
                (positions[prime.id], positions[sub.id])
                for prime, sub in node.elements()
            )
            program.append(_SddOp("decision", 0, children))
        else:
            raise ValueError(f"Unknown SDD node type: {node}")
    return program


def _wmc(
    program: list[_SddOp],
    literal_weights: dict[int, torch.Tensor],
    batch_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Differentiable weighted model count over a linearized SDD program.

    Iterates the flat, pre-linearized program (see
    _compile_wmc_program()) exactly once per call: no Python recursion,
    no per-call memo dict, and no PySDD API calls -- only +/* tensor
    arithmetic over already-computed positions (each "decision" op's
    children always refer to earlier positions, since the program is
    post-order). Every leaf is a (batch_size,) tensor with
    requires_grad tracing back to whatever predicate produced it, so
    this builds an ordinary autograd graph -- no custom backward
    needed.

    Never calls PySDD's own WmcManager.propagate(): that is a
    non-differentiable, non-batched C routine, reserved for test-only
    cross-checks (see SEMANTIC_LOSS_DESIGN.md Section 4.2, 8).

    Args:
        program: Linearized SDD program from _compile_wmc_program().
        literal_weights: Dict mapping SDD variable index to its
            (batch_size,) probability tensor (P(atom=True)).
        batch_shape: Shape to use for true()/false() leaf tensors.
        dtype: Dtype to use for true()/false() leaf tensors.
        device: Device to use for true()/false() leaf tensors.

    Returns:
        Tensor of shape batch_shape with values in [0, 1].
    """
    values: list[torch.Tensor] = []
    for op in program:
        if op.kind == "false":
            values.append(
                torch.zeros(batch_shape, dtype=dtype, device=device)
            )
        elif op.kind == "true":
            values.append(
                torch.ones(batch_shape, dtype=dtype, device=device)
            )
        elif op.kind == "literal":
            prob = literal_weights[abs(op.literal)]
            values.append(prob if op.literal > 0 else (1.0 - prob))
        else:  # "decision"
            total = torch.zeros(batch_shape, dtype=dtype, device=device)
            for prime_pos, sub_pos in op.children:
                total = total + values[prime_pos] * values[sub_pos]
            values.append(total)
    return values[-1]
