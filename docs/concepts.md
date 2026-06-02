# Core Concepts

## Symbols and Variables

**Symbols** represent named predicates (functions that map inputs to truth values in [0, 1]):

```python
from pysignet import Symbol, Variable, And, Or, Not, Implies, Equivalent

# Create predicate symbols
P, Q, R = Symbol("P Q R")

# Create FOL variables (placeholders bound to tensors at evaluation time)
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
values in [0, 1].

### Unary predicates

The simplest case: one variable, one model, scalar output.

```python
P = Symbol("P")
X = Variable("X")
expr = P(X)

model = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())
predicates = {"P": model}
compiled = compile_logic(expr, predicates)
compiled(X=x_tensor)  # model(x_tensor), shape: (batch,)
```

### Multiclass predicates and the class-selector pattern

When an `nn.Module` produces a vector of class probabilities, the *last*
argument in the predicate application acts as a **class selector** that indexes
into that output vector. It is not a second model input.

```python
Digit = Symbol("Digit")
X, Y = Variable("X Y")
expr = Digit(X, Y)  # Y selects which class probability to use

# model takes X, outputs (batch, 10) class probabilities
model = nn.Sequential(nn.Linear(784, 10), nn.Softmax(dim=-1))
predicates = {"Digit": model}

# Passing Y=labels picks output[:, labels[i]] for each batch element i
compiled = compile_logic(expr, predicates)
compiled(X=images, Y=labels)  # model(images)[:, labels]
```

You can also use a constant as the class selector:

```python
expr = Digit(X, 3)  # "X should be classified as digit 3"
compiled(X=images)  # model(images)[:, 3]
```

### Multi-input predicates (binary relations)

When a model takes **two or more input tensors** -- such as a similarity
model `f(x1, x2)` -- it does not fit the single-input `nn.Module` pattern above.
Register it as a **lambda** instead. The lambda is treated as a plain callable:
pysignet passes the predicate arguments to it in the order they appear in the
expression, and thresholds the scalar output at 0.5.

```python
X1, X2 = Variable("X1 X2")
Similar = Symbol("Similar")
expr = Equivalent(Similar(X1, X2), Similar(X2, X1))

# similarity_model.forward(x1, x2) takes two inputs
predicates = {"Similar": lambda x1, x2: similarity_model(x1, x2)}

# pysignet calls lambda(x1, x2) for Similar(X1, X2)
# and lambda(x2, x1) for Similar(X2, X1)
compiled = compile_logic(expr, predicates)
compiled(X1=batch1, X2=batch2)
```

Passing `similarity_model` directly (without a lambda) would cause pysignet to
treat it as a single-input multiclass classifier and apply softmax over the
wrong dimension. The lambda wrapper avoids this.

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
