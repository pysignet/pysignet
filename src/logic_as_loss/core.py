"""
Core functionality for converting SymPy logic expressions to PyTorch loss functions.
"""

import torch
import sympy as sp
from typing import Callable, Dict, Union, Any
from .tnorms import TNorm, ProductTNorm


class Predicate:
    """
    Wrapper for a predicate that can be either a PyTorch model or a deterministic function.
    
    Args:
        name: Name of the predicate (must match SymPy symbol)
        func: Either a torch.nn.Module or a callable that takes inputs and returns [0, 1] values
        is_model: Whether the function is a trainable model (default: auto-detect)
    """
    
    def __init__(
        self, 
        name: str, 
        func: Union[torch.nn.Module, Callable],
        is_model: bool = None
    ):
        self.name = name
        self.func = func
        
        # Auto-detect if it's a model
        if is_model is None:
            self.is_model = isinstance(func, torch.nn.Module)
        else:
            self.is_model = is_model
    
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Evaluate the predicate, ensuring output is in [0, 1]."""
        result = self.func(*args, **kwargs)
        
        # Ensure result is a tensor
        if not isinstance(result, torch.Tensor):
            result = torch.tensor(result, dtype=torch.float32)
        
        # Clamp to [0, 1] for safety
        return torch.clamp(result, 0.0, 1.0)


class LogicLoss:
    """
    Main class for converting SymPy predicate logic expressions into differentiable loss functions.
    
    Args:
        expression: SymPy logic expression (e.g., And(P, Or(Q, Not(R))))
        predicates: Dictionary mapping predicate names to Predicate objects
        tnorm: T-norm to use for relaxation (default: ProductTNorm)
    
    Example:
        >>> import sympy as sp
        >>> import torch.nn as nn
        >>> 
        >>> # Define logic expression
        >>> P, Q, R = sp.symbols('P Q R')
        >>> expr = sp.And(P, sp.Or(Q, sp.Not(R)))
        >>> 
        >>> # Define predicates
        >>> model_P = nn.Sequential(nn.Linear(10, 5), nn.Sigmoid())
        >>> predicates = {
        >>>     'P': Predicate('P', model_P),
        >>>     'Q': Predicate('Q', lambda x: (x > 0).float()),
        >>>     'R': Predicate('R', lambda x: torch.sigmoid(x.sum(dim=-1)))
        >>> }
        >>> 
        >>> # Create loss function
        >>> logic_loss = LogicLoss(expr, predicates)
        >>> 
        >>> # Compute loss on batch
        >>> x = torch.randn(32, 10)  # batch_size=32, features=10
        >>> satisfaction = logic_loss(x)  # Returns tensor of shape (32,)
        >>> loss = 1.0 - satisfaction.mean()  # Convert to loss (higher = worse)
    """
    
    def __init__(
        self, 
        expression: sp.Basic,
        predicates: Dict[str, Predicate],
        tnorm: TNorm = None
    ):
        self.expression = expression
        self.predicates = predicates
        self.tnorm = tnorm or ProductTNorm()
        
        # Verify all symbols in expression have corresponding predicates
        symbols = self._extract_predicate_symbols(expression)
        missing = symbols - set(predicates.keys())
        if missing:
            raise ValueError(f"Missing predicates for symbols: {missing}")
    
    def _extract_predicate_symbols(self, expr: sp.Basic) -> set:
        """Extract all predicate symbols from a SymPy expression."""
        if isinstance(expr, sp.Symbol):
            return {str(expr)}
        
        symbols = set()
        for arg in expr.args:
            symbols.update(self._extract_predicate_symbols(arg))
        return symbols
    
    def _evaluate_expression(
        self, 
        expr: sp.Basic, 
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Recursively evaluate a SymPy expression using t-norm relaxations.
        
        Args:
            expr: SymPy expression to evaluate
            inputs: Either a single tensor (passed to all predicates) or dict of tensors
        
        Returns:
            Tensor of shape (batch_size,) with values in [0, 1]
        """
        # Base case: predicate symbol
        if isinstance(expr, sp.Symbol):
            pred_name = str(expr)
            predicate = self.predicates[pred_name]
            
            # Get input for this predicate
            if isinstance(inputs, dict):
                pred_input = inputs.get(pred_name, inputs.get('default'))
            else:
                pred_input = inputs
            
            return predicate(pred_input)
        
        # Boolean constant
        if expr == sp.true:
            # Return tensor of ones with appropriate batch size
            if isinstance(inputs, dict):
                sample_input = next(iter(inputs.values()))
            else:
                sample_input = inputs
            batch_size = sample_input.shape[0]
            return torch.ones(batch_size, device=sample_input.device)
        
        if expr == sp.false:
            if isinstance(inputs, dict):
                sample_input = next(iter(inputs.values()))
            else:
                sample_input = inputs
            batch_size = sample_input.shape[0]
            return torch.zeros(batch_size, device=sample_input.device)
        
        # Logical operators
        if isinstance(expr, sp.And):
            # Conjoin all arguments
            result = self._evaluate_expression(expr.args[0], inputs)
            for arg in expr.args[1:]:
                result = self.tnorm.conjunction(
                    result,
                    self._evaluate_expression(arg, inputs)
                )
            return result
        
        if isinstance(expr, sp.Or):
            # Disjoin all arguments
            result = self._evaluate_expression(expr.args[0], inputs)
            for arg in expr.args[1:]:
                result = self.tnorm.disjunction(
                    result,
                    self._evaluate_expression(arg, inputs)
                )
            return result
        
        if isinstance(expr, sp.Not):
            return self.tnorm.negation(
                self._evaluate_expression(expr.args[0], inputs)
            )
        
        if isinstance(expr, sp.Implies):
            return self.tnorm.implication(
                self._evaluate_expression(expr.args[0], inputs),
                self._evaluate_expression(expr.args[1], inputs)
            )
        
        if isinstance(expr, sp.Equivalent):
            return self.tnorm.equivalence(
                self._evaluate_expression(expr.args[0], inputs),
                self._evaluate_expression(expr.args[1], inputs)
            )
        
        # Universal quantifier (forall) - interpret as conjunction over batch
        if isinstance(expr, sp.logic.boolalg.ForAll):
            # This is a placeholder - proper handling would require more context
            inner_expr = expr.args[-1]  # Last arg is the expression
            return self._evaluate_expression(inner_expr, inputs)
        
        # Existential quantifier (exists) - interpret as disjunction over batch
        if isinstance(expr, sp.logic.boolalg.Exists):
            inner_expr = expr.args[-1]
            return self._evaluate_expression(inner_expr, inputs)
        
        raise ValueError(f"Unsupported expression type: {type(expr)}")
    
    def __call__(
        self, 
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Evaluate the logic expression on a batch of inputs.
        
        Args:
            inputs: Either a single tensor of shape (batch_size, ...) passed to all predicates,
                   or a dict mapping predicate names to their specific inputs
        
        Returns:
            Tensor of shape (batch_size,) representing satisfaction degree in [0, 1]
            Higher values = better satisfaction of the logical constraint
        """
        return self._evaluate_expression(self.expression, inputs)
    
    def loss(
        self, 
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute loss based on logical constraint violation.
        
        Args:
            inputs: Inputs for predicates
            reduction: 'mean', 'sum', or 'none'
        
        Returns:
            Loss value (lower = better satisfaction)
        """
        satisfaction = self(inputs)
        loss_values = 1.0 - satisfaction  # Convert to loss
        
        if reduction == 'mean':
            return loss_values.mean()
        elif reduction == 'sum':
            return loss_values.sum()
        elif reduction == 'none':
            return loss_values
        else:
            raise ValueError(f"Unknown reduction: {reduction}")
    
    def get_trainable_parameters(self):
        """Get all trainable parameters from model-based predicates."""
        params = []
        for pred in self.predicates.values():
            if pred.is_model and hasattr(pred.func, 'parameters'):
                params.extend(pred.func.parameters())
        return params
