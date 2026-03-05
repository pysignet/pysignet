# pysignet

![pysignet logo](assets/pysignet-logo.png)

[![GitHub](https://img.shields.io/badge/GitHub-pysignet%2Fpysignet-181717?logo=github&logoColor=white)](https://github.com/pysignet/pysignet)
[![CI](https://github.com/pysignet/pysignet/actions/workflows/ci.yml/badge.svg)](https://github.com/pysignet/pysignet/actions/workflows/ci.yml)

pysignet is a PyTorch library that converts symbolic predicate logic expressions
(written in SymPy notation) into differentiable loss functions, enabling you to
train neural networks with logical constraints using First-Order Logic (FOL). It
bridges symbolic reasoning and gradient-based optimization so that logical rules
like implication, mutual exclusion, or quantified constraints become training
signals.

## Quick Start

```python
import torch
import torch.nn as nn
from pysignet import Symbol, Variable, Implies, logic_to_loss

# Define predicate symbols and FOL variables
P, Q = Symbol("P Q")
X = Variable("X")

# "For all inputs X: if P(X) then Q(X)"
expr = Implies(P(X), Q(X))

# Map symbols to neural network models
model_p = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())
model_q = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())

predicates = {
    "P": lambda x: model_p(x).squeeze(-1),
    "Q": lambda x: model_q(x).squeeze(-1),
}

# Compile to a loss function
logic_loss = logic_to_loss(expr, predicates)

# Training loop
x = torch.randn(32, 10)
loss = logic_loss.loss(X=x)   # Bind X to x to get a scalar loss
loss.backward()               # Gradients flow to both models
```

## Installation

```bash
pip install pysignet
```

Or with Poetry:

```bash
poetry add pysignet
```

## Next Steps

- [Core Concepts](concepts.md): Symbols, Variables, Predicates, T-Norms, and Quantifiers
- [API Reference](api.md): Full API documentation auto-generated from docstrings
- [Custom Compilers](custom-compilers.md): Implement your own logic compilation strategy
- [Notebooks](https://github.com/pysignet/pysignet/tree/main/notebooks): Interactive examples on GitHub
