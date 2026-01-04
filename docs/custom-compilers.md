# Implementing Custom Compilers

This guide explains how to implement custom logic compilation strategies by extending the `LogicCompiler` base class.

## Overview

The `LogicCompiler` base class provides all compiler-agnostic functionality, allowing you to focus solely on defining the semantics of logical operators. Implementing a new compiler typically requires only 150-200 lines of code.

## What the Base Class Provides

The `LogicCompiler` base class handles:

1. **Predicate Management**
   - Automatic wrapping of callables in `Predicate` objects
   - Validation that all symbols in expressions have corresponding predicates
   - Assignment of names to predicates

2. **First-Order Logic Support**
   - Quantifier expansion (`ForAll`, `Exists`)
   - Domain size safeguards (configurable limits)
   - Arity validation for multi-variable predicates
   - Handling of mixed variable/constant arguments like `P(X, Y, 0)`

3. **Predicate Evaluation**
   - Routing based on number of free variables (nullary/unary/multi-ary)
   - Input validation and error messages
   - EvaluationContext caching to avoid redundant forward passes
   - Output channel selection for multi-class predicates
   - Boolean constant evaluation (`sp.true`, `sp.false`)

## What You Need to Implement

Your custom compiler must:

1. **Inherit from `LogicCompiler`**
   ```python
   from pysignet.compilation import LogicCompiler

   class MyCompiler(LogicCompiler):
       pass
   ```

2. **Implement `compile()` method**
   - Call base class validation and preprocessing
   - Return a callable that evaluates the expression

3. **Implement `_evaluate_expression()` method**
   - Define semantics for logical operators (AND, OR, NOT, IMPLIES, EQUIVALENT)
   - Use base class methods for predicate evaluation
   - Recursively evaluate subexpressions

## Complete Example: Linear Threshold Unit Compiler

Here's a complete implementation of a compiler that uses linear threshold units:

```python
from typing import Callable, Dict, Union

import sympy as sp
import torch

from pysignet.compilation import LogicCompiler
from pysignet.predicate import Predicate
from pysignet.context import EvaluationContext
from pysignet.multiclass import PredicateApplication


class LinearThresholdUnitCompiler(LogicCompiler):
    """Compiles logic expressions using linear threshold units.

    This compiler represents logical operations as linear threshold units:
    - Conjunction of n literals: sgn(sum(literals) - (n - 0.5))
    - Disjunction of n literals: sgn(sum(literals) - 0.5)
    - Negation: 1 - literal

    Args:
        mode: 'soft' (sigmoid, differentiable) or 'hard' (sign, non-differentiable)
    """

    def __init__(self, mode: str = 'soft') -> None:
        """Initialize LinearThresholdUnitCompiler.

        Args:
            mode: 'soft' for sigmoid or 'hard' for sign function

        Raises:
            ValueError: If mode is not 'soft' or 'hard'
        """
        if mode not in ('soft', 'hard'):
            raise ValueError(f"mode must be 'soft' or 'hard', got '{mode}'")
        self.mode = mode

    def compile(
        self,
        expr: sp.Basic,
        predicates: Dict[str, Predicate]
    ) -> Callable[[Union[torch.Tensor, Dict[str, torch.Tensor]]], torch.Tensor]:
        """Compile a logic expression into a differentiable callable.

        Args:
            expr: SymPy logic expression
            predicates: Dict mapping predicate names to Predicate objects or callables

        Returns:
            Callable that takes inputs and returns satisfaction tensor
        """
        # Use base class for all validation and preprocessing
        wrapped_predicates = self._wrap_and_validate_predicates(expr, predicates)
        expanded_expr = self._expand_quantifiers(expr)

        # Return a closure that evaluates the expression
        def compiled_logic(
            inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]
        ) -> torch.Tensor:
            """Evaluate compiled logic expression."""
            ctx = EvaluationContext()
            return self._evaluate_expression(
                expanded_expr, inputs, wrapped_predicates, ctx
            )

        return compiled_logic

    def _evaluate_expression(
        self,
        expr: sp.Basic,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        predicates: Dict[str, Predicate],
        ctx: EvaluationContext
    ) -> torch.Tensor:
        """Recursively evaluate SymPy expression using LTU operations.

        Args:
            expr: SymPy expression to evaluate
            inputs: Single tensor or dict of tensors
            predicates: Dict of predicates
            ctx: Evaluation context for caching

        Returns:
            Tensor of shape (batch_size,) with values in [0, 1]
        """
        # Base cases: use base class methods for predicate evaluation
        if isinstance(expr, PredicateApplication):
            return self._evaluate_predicate_application(
                expr, inputs, predicates, ctx
            )

        if isinstance(expr, sp.Symbol):
            return self._evaluate_symbol(expr, inputs, predicates, ctx)

        if expr in (sp.true, sp.false):
            return self._evaluate_boolean_constant(expr, inputs)

        # Logical operators: define YOUR semantics here
        if isinstance(expr, sp.Not):
            # Negation: 1 - x
            return 1.0 - self._evaluate_expression(
                expr.args[0], inputs, predicates, ctx
            )

        if isinstance(expr, sp.And):
            # Conjunction: threshold(sum(literals) - (n - 0.5))
            literals = [
                self._evaluate_expression(arg, inputs, predicates, ctx)
                for arg in expr.args
            ]

            summed = torch.stack(literals, dim=0).sum(dim=0)
            threshold = len(literals) - 0.5

            if self.mode == 'soft':
                # Differentiable: sigmoid(k * (sum - threshold))
                return torch.sigmoid(10.0 * (summed - threshold))
            else:
                # Non-differentiable: sign(sum - threshold)
                return ((summed - threshold) >= 0).float()

        if isinstance(expr, sp.Or):
            # Disjunction: threshold(sum(literals) - 0.5)
            literals = [
                self._evaluate_expression(arg, inputs, predicates, ctx)
                for arg in expr.args
            ]

            summed = torch.stack(literals, dim=0).sum(dim=0)
            threshold = 0.5

            if self.mode == 'soft':
                return torch.sigmoid(10.0 * (summed - threshold))
            else:
                return ((summed - threshold) >= 0).float()

        if isinstance(expr, sp.Implies):
            # Implication: (NOT L) OR R
            not_lhs = sp.Not(expr.args[0])
            rhs = expr.args[1]
            return self._evaluate_expression(
                sp.Or(not_lhs, rhs), inputs, predicates, ctx
            )

        if isinstance(expr, sp.Equivalent):
            # Equivalence: (L => R) AND (R => L)
            lhs, rhs = expr.args[0], expr.args[1]
            forward = sp.Implies(lhs, rhs)
            backward = sp.Implies(rhs, lhs)
            return self._evaluate_expression(
                sp.And(forward, backward), inputs, predicates, ctx
            )

        raise ValueError(f"Unsupported expression type: {type(expr)}")
```

## Usage Example

```python
import torch
import sympy as sp
from pysignet import Symbol

# Create your custom compiler
compiler = LinearThresholdUnitCompiler(mode='soft')

# Define logic expression
P, Q = Symbol("P Q")
expr = sp.And(P, Q)

# Define predicates
predicates = {
    "P": lambda x: torch.sigmoid(x.sum(dim=-1)),
    "Q": lambda x: torch.sigmoid(x.mean(dim=-1))
}

# Compile
compiled = compiler.compile(expr, predicates)

# Evaluate
x = torch.randn(10, 5)
result = compiled(x)  # Shape: (10,), values in [0, 1]
```

## Base Class Methods You Should Use

### Validation and Preprocessing

```python
# In your compile() method:
wrapped_predicates = self._wrap_and_validate_predicates(expr, predicates)
expanded_expr = self._expand_quantifiers(expr)
```

These methods:
- Wrap callables in `Predicate` objects
- Validate that all symbols have predicates
- Validate predicate arity
- Expand quantifiers (`ForAll`, `Exists`)
- Check domain size limits

### Predicate Evaluation

```python
# In your _evaluate_expression() method:

# For PredicateApplication (e.g., P(X, Y, 0))
if isinstance(expr, PredicateApplication):
    return self._evaluate_predicate_application(expr, inputs, predicates, ctx)

# For nullary predicates (e.g., P)
if isinstance(expr, sp.Symbol):
    return self._evaluate_symbol(expr, inputs, predicates, ctx)

# For boolean constants (sp.true, sp.false)
if expr in (sp.true, sp.false):
    return self._evaluate_boolean_constant(expr, inputs)
```

These methods handle:
- Free variable extraction
- Input routing (single tensor vs dict)
- Multi-ary predicates
- Caching via `EvaluationContext`
- Output channel selection
- Error messages

## Best Practices

### 1. Always Use Base Class Methods

```python
# GOOD: Use base class for predicate evaluation
if isinstance(expr, PredicateApplication):
    return self._evaluate_predicate_application(expr, inputs, predicates, ctx)

# BAD: Reimplementing predicate logic yourself
if isinstance(expr, PredicateApplication):
    # ... 100 lines of manual handling
```

### 2. Focus on Logical Operator Semantics

Your compiler should only define the semantics of AND, OR, NOT, etc. Everything else is handled by the base class.

```python
# Your compiler focuses on THIS:
if isinstance(expr, sp.And):
    # Define what AND means in your system
    literals = [self._evaluate_expression(arg, ...) for arg in expr.args]
    return your_and_semantics(literals)
```

### 3. Use EvaluationContext for Caching

The base class predicate evaluation methods automatically use `ctx` for caching. Make sure to pass it through:

```python
def _evaluate_expression(self, expr, inputs, predicates, ctx):
    # Pass ctx to base class methods
    return self._evaluate_predicate_application(expr, inputs, predicates, ctx)
```

### 4. Handle All Logical Operators

At minimum, implement:
- `sp.Not` (negation)
- `sp.And` (conjunction)
- `sp.Or` (disjunction)
- `sp.Implies` (implication)
- `sp.Equivalent` (equivalence)

You can implement `Implies` and `Equivalent` in terms of other operators, or provide custom semantics.

### 5. Consider Differentiability

If you want gradients to flow through your logic:
- Use differentiable operations (e.g., `sigmoid` instead of `sign`)
- Avoid operations that break gradient flow (e.g., `.detach()`, comparisons)

### 6. Validate Constructor Arguments

```python
def __init__(self, mode: str = 'soft'):
    if mode not in ('soft', 'hard'):
        raise ValueError(f"mode must be 'soft' or 'hard', got '{mode}'")
    self.mode = mode
```

## Domain Size Configuration

The base class provides configurable domain size limits:

```python
class MyCompiler(LogicCompiler):
    # Override class attributes to change limits
    MAX_DOMAIN_SIZE = 5000  # Raise error above this
    WARN_DOMAIN_SIZE = 500  # Warn above this
```

These limits prevent accidental expansion of huge quantified domains.

## Testing Your Compiler

Create comprehensive tests covering:

1. **Basic operations**: AND, OR, NOT with simple predicates
2. **Implication and equivalence**
3. **Quantifiers**: `ForAll`, `Exists` if supported
4. **Boolean constants**: `sp.true`, `sp.false`
5. **Edge cases**: Empty batches, single elements, nested expressions
6. **Gradient flow**: If differentiable, verify gradients propagate

Example test structure:

```python
import pytest
import torch
from pysignet import Symbol

def test_basic_and():
    """Test AND operation."""
    compiler = MyCompiler()
    P, Q = Symbol("P Q")
    expr = sp.And(P, Q)

    predicates = {
        "P": lambda x: torch.full((x.shape[0],), 0.8),
        "Q": lambda x: torch.full((x.shape[0],), 0.9)
    }

    compiled = compiler.compile(expr, predicates)
    x = torch.randn(5, 10)
    result = compiled(x)

    assert result.shape == (5,)
    assert torch.all((result >= 0) & (result <= 1))
```

## Comparison: T-norm vs LTU Compilers

The library includes two built-in compilers:

**TNormCompiler** (continuous relaxations):
- AND: `a * b` (Product) or `max(0, a+b-1)` (Lukasiewicz)
- OR: `a + b - a*b` (Product) or `min(1, a+b)` (Lukasiewicz)
- NOT: `1 - a`

**LinearThresholdUnitCompiler** (threshold functions):
- AND: `sigmoid(10 * (sum(literals) - (n - 0.5)))`
- OR: `sigmoid(10 * (sum(literals) - 0.5))`
- NOT: `1 - a`

Both use the same base class infrastructure, differing only in operator semantics.

## Summary

Implementing a custom compiler requires:
1. Inherit from `LogicCompiler`
2. Call base class methods in `compile()` for validation/preprocessing
3. Implement `_evaluate_expression()` with your operator semantics
4. Use base class methods for predicate evaluation
5. Test thoroughly

The base class handles all the complexity of predicate management, validation, FOL support, and caching. You focus solely on defining what AND, OR, NOT, etc. mean in your system.
