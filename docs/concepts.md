# Core Concepts

## Symbols and Variables

**Symbols** represent named predicates — functions that map inputs to truth values in [0, 1]:

```python
from pysignet import Symbol, Variable, And, Or, Not, Implies, Equivalent

# Create predicate symbols
P, Q, R = Symbol("P Q R")

# Create FOL variables — placeholders bound to tensors at evaluation time
X, Y = Variable("X Y")
```

**Variables** are bound using keyword arguments at evaluation time:

```python
logic_loss.satisfaction(X=x_tensor)              # Bind X to x_tensor
logic_loss.satisfaction(X=x_tensor, Y=y_tensor)  # Bind multiple variables
```

Predicate applications take the form `P(X)`, `Similar(X, Y)`, or `Digit(X, 3)`.
Constants like `3` are passed through as-is; only variables are bound to tensors.

## Predicates and Arity

Predicates wrap any callable that outputs a tensor of shape `(batch_size,)` with
values in [0, 1]:

```python
# Unary predicate: P(X) maps each input to [0, 1]
P = Symbol("P")
expr = P(X)

# Binary predicate with constant: Digit(X, label)
Digit = Symbol("Digit")
expr = Digit(X, 3)  # "X is digit 3"

# Multi-class classifier — model returns (batch, 10), class index selects output
model = nn.Sequential(nn.Linear(784, 10), nn.Softmax(dim=-1))
predicates = {"Digit": model}
```

For binary relations, use two separate variables:

```python
X1, X2 = Variable("X1 X2")
Similar = Symbol("Similar")
expr = Equivalent(Similar(X1, X2), Similar(X2, X1))  # Symmetry

predicates = {
    "Similar": lambda x1, x2: similarity_model(
        torch.cat([x1, x2], dim=-1)
    ).squeeze(-1)
}
logic_loss.satisfaction(X1=batch1, X2=batch2)
```

## Domain Quantifiers

Use `ForAll` and `Exists` to quantify over finite domains:

```python
from pysignet import ForAll, Exists  # or: from pysignet.logic import ForAll, Exists

X, Y = Variable("X Y")
Digit, Even = Symbol("Digit Even")

# ForAll expands to a conjunction
# "For digits in {0, 2, 4}: if X is classified as that digit, then X is even"
expr = ForAll(Y, [0, 2, 4], Implies(Digit(X, Y), Even(X)))
# Expands to:
#   Implies(Digit(X,0), Even(X)) AND Implies(Digit(X,2), Even(X)) AND Implies(Digit(X,4), Even(X))

# Exists expands to a disjunction
# "X is classified as some digit 0-9"
expr = Exists(Y, range(10), Digit(X, Y))
# Expands to:
#   Digit(X,0) OR Digit(X,1) OR ... OR Digit(X,9)
```

## Batch Quantification

`logic_to_loss` returns a `LogicLoss` that controls how the batch dimension is
reduced via the `quantify` parameter:

```python
from pysignet import logic_to_loss

logic_loss = logic_to_loss(expr, predicates)
x = torch.randn(32, 10)

# Universal quantification (default): ALL batch elements must satisfy
satisfaction = logic_loss.satisfaction(X=x, quantify='forall')  # Scalar

# Existential quantification: AT LEAST ONE element must satisfy
satisfaction = logic_loss.satisfaction(X=x, quantify='exists')  # Scalar

# No quantification: per-element satisfaction
satisfaction = logic_loss.satisfaction(X=x, quantify='none')  # Shape: (32,)
```

Use `compile_logic` when you only need per-batch results (no quantification wrapper):

```python
from pysignet import compile_logic

compiled = compile_logic(expr, predicates)
per_batch = compiled(X=x)  # Shape: (32,) always
```

## T-Norms

T-norms are continuous relaxations of logical operators that map truth values in
[0, 1] to truth values in [0, 1] while preserving differentiability:

| Logic Op  | Product (RProduct) | Lukasiewicz        | Godel          |
|-----------|--------------------|--------------------|----------------|
| AND       | a * b              | max(0, a+b-1)      | min(a, b)      |
| OR        | a+b-a*b            | min(1, a+b)        | max(a, b)      |
| NOT       | 1-a                | 1-a                | 1-a            |
| IMPLIES   | min(1, b/a)        | min(1, 1-a+b)      | max(1-a, b)    |

The default is `MixedTNorm`, which uses `GodelTNorm` for high-arity conjunctions
(arity > 4) and `RProductTNorm` otherwise to balance gradient quality and
numerical stability.

```python
from pysignet.tnorms import RProductTNorm, LukasiewiczTNorm, GodelTNorm

# Specify a t-norm explicitly
logic_loss = logic_to_loss(expr, predicates, tnorm=LukasiewiczTNorm())
```
