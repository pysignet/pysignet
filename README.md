# pysignet: Logic-as-Loss for Neural Networks

[![CI](https://github.com/pysignet/pysignet/actions/workflows/ci.yml/badge.svg)](https://github.com/pysignet/pysignet/actions/workflows/ci.yml)

![Pysignet logo](docs/assets/pysignet-logo-full.png)

pysignet is a PyTorch library that converts symbolic predicate logic expressions
(written in SymPy notation) into differentiable loss functions, enabling you to
train neural networks with logical constraints using First-Order Logic (FOL). It
bridges symbolic reasoning and gradient-based optimization so that logical rules
like implication, mutual exclusion, or quantified constraints become training
signals.

**Documentation:** [pysignet.github.io](https://pysignet.github.io)

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

## Key Features

- **First-Order Logic**: Variables, predicates with arguments, quantifiers
- **Domain Quantifiers**: `ForAll` and `Exists` over finite domains
- **Flexible Predicates**: Neural networks or deterministic functions
- **Multiple T-Norms**: Product, Lukasiewicz, Godel, and Mixed t-norms
- **Full PyTorch Integration**: Gradients flow through all operations

## Learn More

- [Core Concepts](https://pysignet.github.io/concepts/): Symbols, Variables, Predicates, T-Norms, and Quantifiers
- [API Reference](https://pysignet.github.io/api/): Full API documentation
- [Custom Compilers](https://pysignet.github.io/custom-compilers/): Implement your own logic compilation strategy
- [Notebooks](https://github.com/pysignet/pysignet/tree/main/notebooks): Interactive examples

## Development Setup

```bash
git clone https://github.com/pysignet/pysignet.git
cd pysignet
poetry install
poetry run pytest tests/
```

The pre-commit hook (already configured) runs tests, type checking, and linting
before each commit. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## License

MIT License
