# Logic-as-loss: Project summary

## What is this project?

A PyTorch library that converts symbolic predicate logic expressions (written in
SymPy) into differentiable loss functions. This enables training neural networks
with logical constraints.

This library bridges symbolic logic (SymPy) with differentiable optimization
(PyTorch). It allows you to:

- Express logical constraints symbolically using SymPy
- Automatically convert them to differentiable loss functions
- Train neural networks to satisfy logical constraints
- Handle batched operations efficiently
- Use predicates as either PyTorch models or deterministic functions


## Desiderata

1. Efficient batching: All operations should be natively vectorized for batch
   processing.

   ```python
   x = torch.randn(1000, 10)
   satisfaction = compiled_logic(x) # shape: (1000, )
   ```

2. All operations should be sub-differentiable

   ```python
   loss = compiled_logic.loss(inputs)
   loss.backward()
   ```

   After the last step, gradients will flow through the entire logic expression
   to all models that allow for gradients

3. Flexible input handling: The inputs can be single examples (one image, one
   document, etc) or groups of them. The input handling flexibility comes from
   the mapping from symbols to predicates. The predicates can work with subsets
   of the provided inputs.

   ```python
   x = {
       "image": image_data,
       "text": text_data,
       "sensor": sensor_data
   }

   satisfaction = compiled_logic(x)
   ```

   At the last step, the appropriate models will get the appropriate elements of
   `x` because of how the predicates are mapped to symbols.




## Key Features

- **Symbolic Logic**: Write constraints using familiar logic operators in sympy
  (AND, OR, NOT, IMPLIES, etc.)
- **Flexible Predicates**: Predicates can be neural networks or deterministic
  functions of inputs
- **Batching Support**: All operations support batched inputs for efficient
  training
- **Multiple T-Norms**: Choose from Product, Łukasiewicz, or Gödel t-norms
- **Full PyTorch Integration**: Gradients flow through all operations, use any
  PyTorch features
- **Per-Predicate Inputs**: Different predicates can receive different inputs


## Quick example and usage pattern

The usage pattern involves the following four steps:

1. Define logic with SymPy (symbols and the logic expression)
2. Map each symbol in SymPy to predicates, which could be model calls or
   input-dependent functions (or combinations of them)
3. Create the logic loss compiler
4. Train


```python
import torch
import torch.nn as nn
import sympy as sp
from logic_as_loss import LogicCompiler, Predicate

# Define logic expression. First define the symbols
P, Q, R = sp.symbols('P Q R')

# the transitivity property
expr = sp.Implies(sp.And(sp.Implies(P, Q),
                         sp.Implies(Q, R))
                  sp.Implies(P, R))

# Define a model
model = nn.Sequential(nn.Linear(10, 5), nn.Sigmoid())

# Inputs contain two parts indexed by "A" and "B"
predicates = {
    'P': Predicate('P', lambda x: model(x["A"])[0, :]),
    'Q': Predicate('Q', lambda x: (x["B"].sum(dim=-1) > 0).float()),
    'R': Predicate('R', lambda x: model(x["B"][1:, :]))
}

# Create logic compiler
compiled_logic = LogicCompiler(expr, predicates)

# Use in training
x = torch.randn(32, 10)  # batch_size=32
satisfaction = compiled_logic(x)  # Shape: (32,), values in [0, 1]
loss = compiled_logic.loss(x)    # Scalar loss = 1 - mean(satisfaction)

# Backpropagate
loss.backward()
```

## What Makes This Useful

1. Neural-Symbolic AI: Combine neural learning with symbolic reasoning:
```python
# "If person is detected, face must be detected"
expr = sp.Implies(person_detected, face_detected)
```

2. Constrained Learning: Enforce domain knowledge during training:
```python
# "Temperature and pressure must be in safe range"
expr = sp.And(safe_temp, safe_pressure)
```

3. Semi-Supervised Learning: Use logic rules as weak supervision:
```python
# "If labeled positive, confidence should be high"
expr = sp.Implies(is_positive_label, high_confidence)
```

4. Multi-Task Learning: Enforce consistency between tasks:
```python
# "If task1 predicts X, task2 should predict Y"
expr = sp.Implies(task1_predicts_X, task2_predicts_Y)
```


## Core Concepts

### T-Norms

T-norms are continuous relaxations of discrete logical operators:

| Logic Op    | Product T-Norm | Łukasiewicz T-Norm | Gödel T-Norm          |
|-------------|----------------|--------------------|-----------------------|
| AND (∧)     | a × b          | max(0, a+b-1)      | min(a, b)             |
| OR (∨)      | a+b-a×b        | min(1, a+b)        | max(a, b)             |
| NOT (¬)     | 1-a            | 1-a                | 1-a                   |
| IMPLIES (→) | max(1, y/x)    | max(1, 1-x+y)      | [NOT DIFFERENTIABLE!] |

**Product T-Norm** (default): Best for gradient flow, most commonly used in neural-symbolic learning.

**Łukasiewicz T-Norm**: Stricter constraints, good for enforcing hard logical rules.

**Gödel T-Norm**: Most conservative, but can have gradient issues at boundaries.


### Predicates

A `Predicate` wraps a function or model that evaluates to [0, 1]:

```python
# Neural network predicate
model = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())
pred = Predicate('P', lambda x: model(x).squeeze(-1))

# Deterministic function predicate
pred = Predicate('Q', lambda x: (x > 0).float().mean(dim=-1))
```

### Logic Loss:

The `LogicCompiler` class converts SymPy expressions to differentiable functions:

```python
compiler = LogicCompiler(
    expression=expr,         # SymPy logic expression
    predicates=preds,        # Dict of predicates
    tnorm=RProductTNorm()    # Optional: t-norm to use (default: RProductTNorm)
)

# Evaluate satisfaction (higher = better)
satisfaction = compiler(inputs)  # Returns tensor in [0, 1]

# Compute loss (lower = better)
loss = compiler.loss(inputs, reduction='mean')
```

- TODO: Different relaxations require different post-processing.
  - For R-Product/S-ProductTNorm, the loss should be negative log satisfaction
  - For Lukasiewicz, it should just be negative satisfaction
  - For all of them, the post-processing should be specifiable

## Extensions & Future Work

Possible extensions (not yet implemented):
- Quantifiers (∀, ∃) over batch dimensions
- Weighted predicates for importance
- Fuzzy membership functions
- First-order logic support
- Temporal logic operators
- Probabilistic extensions
