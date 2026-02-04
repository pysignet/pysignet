# pysignet: Logic-as-Loss for Neural Networks

[![CI](https://github.com/pysignet/pysignet/actions/workflows/ci.yml/badge.svg)](https://github.com/pysignet/pysignet/actions/workflows/ci.yml)

![Pysignet logo](assets/pysignet-logo.png)

## What is this project?

A PyTorch library that converts symbolic predicate logic expressions (written in
SymPy) into differentiable loss functions. This enables training neural networks
with logical constraints using First-Order Logic (FOL).

This library bridges symbolic logic (SymPy) with differentiable optimization
(PyTorch). It allows you to:

- Express logical constraints symbolically using SymPy with FOL support
- Use variables and quantifiers (ForAll, Exists) over domains
- Automatically convert constraints to differentiable loss functions
- Train neural networks to satisfy logical constraints
- Handle batched operations with explicit quantification semantics

## Quick Example

```python
import torch
import torch.nn as nn
import sympy as sp
from pysignet import Symbol, Variable, logic_to_loss

# Define symbols and variables
P, Q = Symbol("P Q")
X = Variable("X")

# Define a logical expression using FOL
# "For all inputs X, if P(X) then Q(X)"
expr = sp.Implies(P(X), Q(X))

# Define neural network models
model_p = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())
model_q = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())

# Map symbols to models (auto-wrapped as predicates)
predicates = {
    "P": lambda x: model_p(x).squeeze(-1),
    "Q": lambda x: model_q(x).squeeze(-1)
}

# Compile the logic expression for training
logic_loss = logic_to_loss(expr, predicates)

# Use in training
x = torch.randn(32, 10)  # batch of 32 inputs

# Evaluate satisfaction (default: forall quantification over batch)
satisfaction = logic_loss(X=x)  # Scalar in [0, 1]

# Compute loss for training
loss = logic_loss.loss(X=x)  # Scalar loss
loss.backward()  # Gradients flow to both models
```

## Key Features

- **First-Order Logic**: Variables, predicates with arguments, quantifiers
- **Domain Quantifiers**: ForAll and Exists over finite domains
- **Batch Quantification**: Explicit `quantify` parameter ('forall', 'exists', 'none')
- **Flexible Predicates**: Neural networks or deterministic functions
- **Multiple T-Norms**: Product, Lukasiewicz, or Godel t-norms
- **Full PyTorch Integration**: Gradients flow through all operations
- **Numerical Stability**: Log-space computation for large batches

## Installation

```bash
pip install pysignet
```

Or with Poetry:
```bash
poetry add pysignet
```

## Core Concepts

### Symbols and Variables

**Symbols** represent predicates (named neurons):
```python
from pysignet import Symbol, Variable

# Create predicate symbols
P, Q, R = Symbol("P Q R")

# Create variables for FOL
X, Y = Variable("X Y")
```

**Variables** are placeholders bound to tensors at evaluation time:
```python
# P(X) - unary predicate applied to variable X
# Similar(X, Y) - binary predicate comparing two inputs
# Digit(X, 0) - predicate with variable and constant arguments
```

### Predicates and Arity

Predicates map inputs to truth values [0, 1]:

```python
# Unary predicate: P(X) -> [0, 1]
P = Symbol("P")
expr = P(X)  # Property P holds for X

# Binary predicate with constant: Digit(X, label) -> [0, 1]
Digit = Symbol("Digit")
expr = Digit(X, 3)  # "X is digit 3"

# Multi-class classifier (10 outputs, select by index)
model = nn.Sequential(nn.Linear(784, 10), nn.Softmax(dim=-1))
predicates = {"Digit": model}  # Returns (batch, 10), index selects class
```

### Domain Quantifiers

Quantify over finite domains:

```python
from pysignet.logic import ForAll, Exists

X, Y = Variable("X Y")
Digit, Even = Symbol("Digit Even")

# "For digits in 0, 2, 4, if X is classified as that digit, then X is even"
# Expands to: Implies(Digit(X,0), Even(X)) AND Implies(Digit(X,2), Even(X)) AND
Implies(Digit(X,2), Even(X))
expr = ForAll(Y, [0, 2, 4], Implies(Digit(X, Y), Even(X)))

# "X is classified as some digit 0-9"
# Expands to: Digit(X,0) OR Digit(X,1) OR ... OR Digit(X,9)
expr = Exists(Y, range(10), Digit(X, Y))
```

### Batch Quantification

Control how batch dimensions are handled with `logic_to_loss`:

```python
logic_loss = logic_to_loss(expr, predicates)
x = torch.randn(32, 10)

# Universal quantification (default): ALL batch elements must satisfy
satisfaction = logic_loss(X=x, quantify='forall')  # Scalar

# Existential quantification: AT LEAST ONE element must satisfy
satisfaction = logic_loss(X=x, quantify='exists')  # Scalar

# No quantification: per-element satisfaction
satisfaction = logic_loss(X=x, quantify='none')  # Shape: (32,)
```

For per-batch results only (without quantification), use `compile_logic`:

```python
compiled = compile_logic(expr, predicates)
per_batch = compiled(X=x)  # Shape: (32,) - always per-batch
```

### T-Norms

T-norms are continuous relaxations of logical operators:

| Logic Op    | Product T-Norm | Lukasiewicz T-Norm | Godel T-Norm |
|-------------|----------------|--------------------|--------------|
| AND         | a * b          | max(0, a+b-1)      | min(a, b)    |
| OR          | a+b-a*b        | min(1, a+b)        | max(a, b)    |
| NOT         | 1-a            | 1-a                | 1-a          |
| IMPLIES     | min(1, b/a)    | min(1, 1-a+b)      | 1 if a<=b    |

```python
# Use different t-norms
logic_loss = compile_logic(expr, predicates, tnorm='rproduct')  # Default
logic_loss = compile_logic(expr, predicates, tnorm='lukasiewicz')
logic_loss = compile_logic(expr, predicates, tnorm='godel')
```

## Examples

### MNIST Digit Classification with Logic

```python
from pysignet import Symbol, Variable, compile_logic
from pysignet.logic import Exists

# "Each image is classified as exactly one digit"
X, Y = Variable("X Y")
Digit = Symbol("Digit")

# At least one class has high confidence
expr = Exists(Y, range(10), Digit(X, Y))

# 10-class classifier
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.Softmax(dim=-1)
)

logic_loss = logic_to_loss(expr, {"Digit": model})

# Training loop
for images, labels in dataloader:
    loss = logic_loss.loss(X=images)
    loss.backward()
    optimizer.step()
```

### Mutual Exclusion Constraint

```python
# "If P(X) is true, then Q(X) must be false"
P, Q = Symbol("P Q")
X = Variable("X")

expr = sp.Implies(P(X), sp.Not(Q(X)))

# For training with loss computation
logic_loss = logic_to_loss(expr, {
    "P": model_p,
    "Q": model_q
})

# For per-batch evaluation only
compiled = compile_logic(expr, {
    "P": model_p,
    "Q": model_q
})
```

### Multi-Input Predicates

```python
# Compare two inputs for similarity
X, Y = Variable("X Y")
Similar = Symbol("Similar")

# Binary predicate taking two inputs
expr = Similar(X, Y)

def similarity_fn(x, y):
    # Concatenate and classify
    combined = torch.cat([x, y], dim=-1)
    return similarity_model(combined).squeeze(-1)

predicates = {"Similar": similarity_fn}

# For per-batch evaluation
compiled = compile_logic(expr, predicates)
x1 = torch.randn(32, 10)
x2 = torch.randn(32, 10)
satisfaction = compiled(X=x1, Y=x2)  # Shape: (32,)

# For training with quantification and loss
logic_loss = logic_to_loss(expr, predicates)
satisfaction = logic_loss(X=x1, Y=x2)  # Scalar (forall)
```

## API Reference

### compile_logic

Returns `CompiledExpression` for per-batch evaluation:

```python
compile_logic(
    expression,           # SymPy logic expression with FOL
    predicates,           # Dict mapping symbol names to callables/models
    tnorm='rproduct'      # T-norm: 'rproduct', 'sproduct', 'lukasiewicz', 'godel'
) -> CompiledExpression
```

### logic_to_loss

Returns `LogicLoss` with quantification and loss methods:

```python
logic_to_loss(
    expression,           # SymPy logic expression with FOL
    predicates,           # Dict mapping symbol names to callables/models
    tnorm='rproduct'      # T-norm: 'rproduct', 'sproduct', 'lukasiewicz', 'godel'
) -> LogicLoss
```

### CompiledExpression

```python
# Evaluation (always per-batch)
satisfaction = compiled(X=x)  # Shape: (batch_size,)

# Partial binding
partial = compiled.partial(X=x)
result = partial(Y=y)
```

### LogicLoss

```python
# Evaluation with quantification
satisfaction = logic_loss(X=x, quantify='forall')  # Scalar [0, 1]
satisfaction = logic_loss(X=x, quantify='exists')  # Scalar [0, 1]
satisfaction = logic_loss(X=x, quantify='none')    # Shape: (batch_size,)
log_sat = logic_loss.log_satisfaction(X=x)         # (-inf, 0]

# Loss computation
loss = logic_loss.loss(X=x, quantify='forall')                    # Scalar
loss = logic_loss.loss(X=x, quantify='none', reduction='mean')    # Mean of per-element
loss = logic_loss.loss(X=x, quantify='none', reduction='sum')     # Sum of per-element
loss = logic_loss.loss(X=x, quantify='none', reduction='none')    # Per-element losses
```

### Quantify Modes

| Mode | Returns | Meaning |
|------|---------|---------|
| `'forall'` | Scalar | ALL batch elements satisfy |
| `'exists'` | Scalar | AT LEAST ONE element satisfies |
| `'none'` | (batch,) | Per-element satisfaction |

## See Also

- `examples/` - Comprehensive example scripts

## License

MIT License
