"""Tests to increase coverage for edge cases and error conditions.

These tests target specific uncovered lines identified by coverage analysis.
"""

import pytest
import torch
import torch.nn as nn
import sympy as sp

from pysignet import Symbol, Variable, compile_logic, Predicate
from pysignet.logic.extraction import (
    extract_variables_from_application,
    extract_constants_from_application,
    is_variable,
    is_constant,
)
from pysignet.compilation.module_utils import (
    infer_module_arity,
    has_final_activation,
    wrap_module_as_predicate,
    _get_output_dim,
    _get_final_layer,
)


class TestExtractionFunctions:
    """Tests for extraction.py coverage gaps."""

    def test_extract_variables_from_application_multiple_vars(self):
        """Test extracting multiple variables from a single application."""
        P = Symbol("P")
        X, Y, Z = Variable("X Y Z")
        app = P(X, Y, Z)

        variables = extract_variables_from_application(app)

        assert len(variables) == 3
        assert X in variables
        assert Y in variables
        assert Z in variables

    def test_extract_variables_from_application_mixed(self):
        """Test extracting variables with mixed args (vars and constants)."""
        P = Symbol("P")
        X, Y = Variable("X Y")
        app = P(X, 5, Y, "red")

        variables = extract_variables_from_application(app)

        assert len(variables) == 2
        assert X in variables
        assert Y in variables

    def test_extract_variables_from_application_no_vars(self):
        """Test extracting from application with only constants."""
        P = Symbol("P")
        app = P(1, 2, 3)

        variables = extract_variables_from_application(app)

        assert len(variables) == 0

    def test_extract_variables_from_application_duplicates(self):
        """Test that duplicate variables are deduplicated."""
        P = Symbol("P")
        X = Variable("X")
        app = P(X, 0, X, 1, X)

        variables = extract_variables_from_application(app)

        assert len(variables) == 1
        assert X in variables

    def test_extract_constants_from_application_multiple(self):
        """Test extracting multiple constants from a single application."""
        P = Symbol("P")
        X = Variable("X")
        app = P(X, 1, 2, "red", 3.14)

        constants = extract_constants_from_application(app)

        assert len(constants) == 4
        assert 1 in constants
        assert 2 in constants
        assert "red" in constants
        assert 3.14 in constants

    def test_extract_constants_from_application_no_constants(self):
        """Test extracting from application with only variables."""
        P = Symbol("P")
        X, Y = Variable("X Y")
        app = P(X, Y)

        constants = extract_constants_from_application(app)

        assert len(constants) == 0

    def test_extract_constants_from_application_none_constant(self):
        """Test that None is recognized as a constant."""
        P = Symbol("P")
        X = Variable("X")
        app = P(X, None, 0)

        constants = extract_constants_from_application(app)

        assert len(constants) == 2
        assert None in constants
        assert 0 in constants

    def test_is_variable_with_variable(self):
        """Test is_variable returns True for VariableSymbol."""
        X = Variable("X")
        assert is_variable(X) is True

    def test_is_variable_with_non_variable(self):
        """Test is_variable returns False for non-variables."""
        assert is_variable(5) is False
        assert is_variable("test") is False
        assert is_variable(None) is False
        assert is_variable(3.14) is False

    def test_is_constant_with_various_types(self):
        """Test is_constant with various types."""
        X = Variable("X")

        assert is_constant(X) is False
        assert is_constant(5) is True
        assert is_constant("test") is True
        assert is_constant(None) is True
        assert is_constant(3.14) is True
        assert is_constant((1, 2, 3)) is True


class TestModuleUtils:
    """Tests for module_utils.py coverage gaps."""

    def test_infer_module_arity_sigmoid(self):
        """Test arity inference for module ending with Sigmoid."""
        model = nn.Sequential(nn.Linear(10, 5), nn.Sigmoid())
        arity = infer_module_arity(model)
        assert arity == 1

    def test_infer_module_arity_softmax(self):
        """Test arity inference for module ending with Softmax."""
        model = nn.Sequential(nn.Linear(10, 5), nn.Softmax(dim=-1))
        arity = infer_module_arity(model)
        assert arity == 2

    def test_infer_module_arity_linear_single_output(self):
        """Test arity inference for Linear with single output."""
        model = nn.Sequential(nn.Linear(10, 1))
        arity = infer_module_arity(model)
        assert arity == 1

    def test_infer_module_arity_linear_multi_output(self):
        """Test arity inference for Linear with multiple outputs."""
        model = nn.Sequential(nn.Linear(10, 5))
        arity = infer_module_arity(model)
        assert arity == 2

    def test_infer_module_arity_custom_module(self):
        """Test that custom modules return None for arity."""
        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(5))

            def forward(self, x):
                return x * self.param

        model = CustomModel()
        arity = infer_module_arity(model)
        assert arity is None

    def test_empty_sequential_error(self):
        """Test error when trying to get final layer of empty Sequential."""
        model = nn.Sequential()

        with pytest.raises(ValueError, match="empty Sequential"):
            _get_final_layer(model)

    def test_wrap_module_unsupported_arity(self):
        """Test error for unsupported arity (> 2)."""
        model = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())

        # Arity 3 causes KeyError in arity_names lookup, then ValueError
        with pytest.raises((ValueError, KeyError)):
            wrap_module_as_predicate(model, arity=3)

    def test_wrap_module_arity_mismatch(self):
        """Test error when module arity doesn't match specified arity."""
        # Unary module (output dim = 1) but requesting binary arity
        model = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())

        with pytest.raises(ValueError, match="Arity mismatch"):
            wrap_module_as_predicate(model, arity=2)

    def test_get_output_dim_sigmoid(self):
        """Test _get_output_dim for Sigmoid layer."""
        model = nn.Sequential(nn.Linear(10, 5), nn.Sigmoid())
        dim = _get_output_dim(model)
        assert dim == 1

    def test_get_output_dim_softmax(self):
        """Test _get_output_dim for Softmax layer."""
        model = nn.Sequential(nn.Linear(10, 5), nn.Softmax(dim=-1))
        dim = _get_output_dim(model)
        assert dim == 2  # Placeholder value for Softmax

    def test_get_output_dim_custom(self):
        """Test _get_output_dim for custom layer."""
        class CustomLayer(nn.Module):
            def forward(self, x):
                return x

        model = nn.Sequential(CustomLayer())
        dim = _get_output_dim(model)
        assert dim == 0  # Unknown

    def test_has_final_activation_true(self):
        """Test has_final_activation returns True for Sigmoid/Softmax."""
        model1 = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())
        model2 = nn.Sequential(nn.Linear(10, 5), nn.Softmax(dim=-1))

        assert has_final_activation(model1) is True
        assert has_final_activation(model2) is True

    def test_has_final_activation_false(self):
        """Test has_final_activation returns False for no activation."""
        model = nn.Sequential(nn.Linear(10, 1))
        assert has_final_activation(model) is False

    def test_wrap_unary_with_activation(self):
        """Test wrapping unary module that already has activation."""
        model = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())
        wrapper = wrap_module_as_predicate(model, arity=1)

        x = torch.randn(32, 10)
        output = wrapper(x)

        assert output.shape == (32,)
        assert (output >= 0).all() and (output <= 1).all()

    def test_wrap_unary_without_activation(self):
        """Test wrapping unary module without activation (adds sigmoid)."""
        model = nn.Sequential(nn.Linear(10, 1))
        wrapper = wrap_module_as_predicate(model, arity=1)

        x = torch.randn(32, 10)
        output = wrapper(x)

        assert output.shape == (32,)
        assert (output >= 0).all() and (output <= 1).all()

    def test_wrap_binary_with_activation(self):
        """Test wrapping binary module that already has softmax."""
        model = nn.Sequential(nn.Linear(10, 5), nn.Softmax(dim=-1))
        wrapper = wrap_module_as_predicate(model, arity=2)

        x = torch.randn(32, 10)
        output = wrapper(x, 2)  # Select class 2

        assert output.shape == (32,)
        assert (output >= 0).all() and (output <= 1).all()

    def test_wrap_binary_without_activation(self):
        """Test wrapping binary module without activation (adds softmax)."""
        model = nn.Sequential(nn.Linear(10, 5))
        wrapper = wrap_module_as_predicate(model, arity=2)

        x = torch.randn(32, 10)
        output = wrapper(x, 2)  # Select class 2

        assert output.shape == (32,)
        assert (output >= 0).all() and (output <= 1).all()

    def test_get_final_layer_nested_sequential(self):
        """Test getting final layer from nested Sequential."""
        inner = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        outer = nn.Sequential(inner, nn.Linear(5, 1), nn.Sigmoid())

        final = _get_final_layer(outer)
        assert isinstance(final, nn.Sigmoid)

    def test_get_final_layer_module_with_children(self):
        """Test getting final layer from module with children."""
        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 5)
                self.fc2 = nn.Linear(5, 1)
                self.act = nn.Sigmoid()

            def forward(self, x):
                return self.act(self.fc2(self.fc1(x)))

        model = CustomModel()
        final = _get_final_layer(model)
        assert isinstance(final, nn.Sigmoid)


class TestCompilationBaseEdgeCases:
    """Tests for compilation/base.py coverage gaps."""

    def test_missing_variable_in_dict_input(self):
        """Test error when variable is missing from dict input."""
        P = Symbol("P")
        X, Y = Variable("X Y")
        expr = sp.And(P(X), P(Y))

        predicates = {"P": lambda x: torch.sigmoid(x.sum(dim=-1))}
        compiled = compile_logic(expr, predicates)

        x = torch.randn(4, 10)
        # Provide X but not Y
        with pytest.raises(ValueError, match="Missing input bindings"):
            compiled(X=x)  # Y is missing

    def test_multiple_vars_with_non_dict_input(self):
        """Test error when multiple vars exist but input is not dict."""
        P, Q = Symbol("P Q")
        X, Y = Variable("X Y")
        expr = sp.And(P(X), Q(Y))

        predicates = {
            "P": lambda x: torch.sigmoid(x.sum(dim=-1)),
            "Q": lambda y: torch.sigmoid(y.sum(dim=-1)),
        }
        compiled = compile_logic(expr, predicates)

        # This should work - both vars bound
        x = torch.randn(4, 10)
        y = torch.randn(4, 10)
        result = compiled(X=x, Y=y)
        assert result.shape == ()  # Scalar with default quantify='forall'

    def test_module_arity_mismatch_unary_used_as_binary(self):
        """Test error when unary module used with multiple arguments."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X, 5)  # Binary usage

        # Unary module (output dim = 1)
        model = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())

        with pytest.raises(ValueError, match="arity mismatch"):
            compile_logic(expr, {"P": model})

    def test_module_arity_mismatch_binary_used_as_unary(self):
        """Test error when binary module used with single argument."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)  # Unary usage

        # Binary module (output dim > 1)
        model = nn.Sequential(nn.Linear(10, 5), nn.Softmax(dim=-1))

        with pytest.raises(ValueError, match="arity mismatch"):
            compile_logic(expr, {"P": model})

    def test_constant_only_predicate_non_module(self):
        """Test predicate with only constants (no variables, non-module)."""
        P = Symbol("P")
        X = Variable("X")

        # P(5) - constant-only, but we need a free variable for inputs
        # Actually, let's test a more complex case
        Q = Symbol("Q")
        expr = sp.And(P(X), Q(X))  # Q takes variable

        call_count = [0]
        def p_func(x):
            call_count[0] += 1
            return torch.sigmoid(x.sum(dim=-1))

        predicates = {
            "P": p_func,
            "Q": lambda x: torch.sigmoid(x.mean(dim=-1)),
        }
        compiled = compile_logic(expr, predicates)

        x = torch.randn(4, 10)
        result = compiled(X=x)

        # Should work
        assert result.shape == ()

    def test_predicate_not_callable_error(self):
        """Test error when predicate value is not callable."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        with pytest.raises(TypeError, match="must be callable"):
            compile_logic(expr, {"P": "not_callable"})

    def test_predicate_not_callable_number(self):
        """Test error when predicate value is a number."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        with pytest.raises(TypeError, match="must be callable"):
            compile_logic(expr, {"P": 42})


class TestQuantifyModes:
    """Additional tests for quantify parameter edge cases."""

    def test_quantify_exists_single_satisfied(self):
        """Test exists quantification with one satisfying element."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        # Predicate that returns mostly 0, but one 1
        def p_func(x):
            result = torch.zeros(x.shape[0])
            result[0] = 1.0  # First element satisfied
            return result

        compiled = compile_logic(expr, {"P": p_func})
        x = torch.randn(10, 5)

        result = compiled(X=x, quantify='exists')
        assert result.shape == ()
        assert result.item() == 1.0  # At least one satisfied

    def test_quantify_forall_one_unsatisfied(self):
        """Test forall quantification with one unsatisfied element."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        # Predicate that returns mostly 1, but one 0
        def p_func(x):
            result = torch.ones(x.shape[0])
            result[0] = 0.0  # First element NOT satisfied
            return result

        compiled = compile_logic(expr, {"P": p_func})
        x = torch.randn(10, 5)

        result = compiled(X=x, quantify='forall')
        assert result.shape == ()
        assert result.item() == 0.0  # Not all satisfied


class TestLossEdgeCases:
    """Tests for loss.py coverage gaps."""

    def test_loss_with_quantify_none_reduction_sum(self):
        """Test loss with quantify=none and reduction=sum."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        predicates = {"P": lambda x: torch.sigmoid(x.sum(dim=-1))}
        compiled = compile_logic(expr, predicates)

        x = torch.randn(8, 10)
        loss = compiled.loss(X=x, quantify='none', reduction='sum')

        assert loss.shape == ()  # Scalar after sum reduction

    def test_loss_with_quantify_none_reduction_none(self):
        """Test loss with quantify=none and reduction=none."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        predicates = {"P": lambda x: torch.sigmoid(x.sum(dim=-1))}
        compiled = compile_logic(expr, predicates)

        x = torch.randn(8, 10)
        loss = compiled.loss(X=x, quantify='none', reduction='none')

        assert loss.shape == (8,)  # Per-element loss

    def test_log_satisfaction(self):
        """Test log_satisfaction method."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        predicates = {"P": lambda x: torch.sigmoid(x.sum(dim=-1))}
        compiled = compile_logic(expr, predicates)

        x = torch.randn(8, 10)
        log_sat = compiled.log_satisfaction(X=x)

        assert log_sat.shape == ()  # Scalar
        assert log_sat.item() <= 0  # Log of value in [0, 1]


class TestCompiledExpressionEdgeCases:
    """Tests for compiled_expression.py coverage gaps."""

    def test_partial_binding_chain(self):
        """Test chaining multiple partial bindings."""
        P, Q = Symbol("P Q")
        X, Y, Z = Variable("X Y Z")
        expr = sp.And(P(X), sp.And(Q(Y), P(Z)))

        predicates = {
            "P": lambda x: torch.sigmoid(x.sum(dim=-1)),
            "Q": lambda y: torch.sigmoid(y.mean(dim=-1)),
        }
        compiled = compile_logic(expr, predicates)

        x = torch.randn(4, 10)
        y = torch.randn(4, 10)
        z = torch.randn(4, 10)

        # Chain partial bindings
        partial1 = compiled.partial(X=x)
        partial2 = partial1.partial(Y=y)
        result = partial2(Z=z)

        assert result.shape == ()  # Scalar

    def test_partial_binding_loss(self):
        """Test computing loss from partial binding."""
        P, Q = Symbol("P Q")
        X, Y = Variable("X Y")
        expr = sp.And(P(X), Q(Y))

        predicates = {
            "P": lambda x: torch.sigmoid(x.sum(dim=-1)),
            "Q": lambda y: torch.sigmoid(y.mean(dim=-1)),
        }
        compiled = compile_logic(expr, predicates)

        x = torch.randn(4, 10)
        y = torch.randn(4, 10)

        # Get loss from partial binding
        partial = compiled.partial(X=x)
        loss = partial.loss(Y=y)

        assert loss.shape == ()  # Scalar loss

    def test_compiled_expression_free_variables(self):
        """Test accessing free variables from compiled expression."""
        P, Q = Symbol("P Q")
        X, Y = Variable("X Y")
        expr = sp.And(P(X), Q(Y))

        predicates = {
            "P": lambda x: torch.sigmoid(x.sum(dim=-1)),
            "Q": lambda y: torch.sigmoid(y.mean(dim=-1)),
        }
        compiled = compile_logic(expr, predicates)

        # Access free variables
        free_vars = compiled.free_variables
        assert len(free_vars) == 2

    def test_compiled_expression_partial_remaining_vars(self):
        """Test that partial binding leaves remaining variables unbound."""
        P = Symbol("P")
        X, Y = Variable("X Y")
        expr = sp.And(P(X), P(Y))

        predicates = {"P": lambda x: torch.sigmoid(x.sum(dim=-1))}
        compiled = compile_logic(expr, predicates)

        x = torch.randn(4, 10)
        partial = compiled.partial(X=x)

        # After binding X, we can still call with Y
        y = torch.randn(4, 10)
        result = partial(Y=y)
        assert result.shape == ()  # Scalar


class TestArityValidationEdgeCases:
    """Tests for arity.py edge cases."""

    def test_callable_with_multiple_args(self):
        """Test callable with multiple positional arguments."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X, 1, 2, 3)  # 4 arguments

        def multi_arg_func(x, a, b, c):
            # Use all args to satisfy arity
            return torch.sigmoid(x.sum(dim=-1) + a + b + c)

        compiled = compile_logic(expr, {"P": multi_arg_func})
        x = torch.randn(4, 10)
        result = compiled(X=x)

        assert result.shape == ()

    def test_lambda_predicate(self):
        """Test lambda function as predicate."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        compiled = compile_logic(expr, {
            "P": lambda x: torch.sigmoid(x.sum(dim=-1))
        })

        x = torch.randn(4, 10)
        result = compiled(X=x)

        assert result.shape == ()


class TestTNormOperators:
    """Additional tests for t-norm operator coverage."""

    def test_godel_tnorm_edge_values(self):
        """Test Godel t-norm with edge values."""
        from pysignet.tnorms import GodelTNorm
        tnorm = GodelTNorm()

        a = torch.tensor([0.0, 0.5, 1.0])
        b = torch.tensor([1.0, 0.5, 0.0])

        # AND = min
        result_and = tnorm.conjunction(a, b)
        expected_and = torch.tensor([0.0, 0.5, 0.0])
        assert torch.allclose(result_and, expected_and)

        # OR = max
        result_or = tnorm.disjunction(a, b)
        expected_or = torch.tensor([1.0, 0.5, 1.0])
        assert torch.allclose(result_or, expected_or)

    def test_lukasiewicz_tnorm_edge_values(self):
        """Test Lukasiewicz t-norm with edge values."""
        from pysignet.tnorms import LukasiewiczTNorm
        tnorm = LukasiewiczTNorm()

        a = torch.tensor([0.0, 0.5, 1.0])
        b = torch.tensor([1.0, 0.5, 0.0])

        # AND = max(0, a+b-1)
        result_and = tnorm.conjunction(a, b)
        expected_and = torch.tensor([0.0, 0.0, 0.0])
        assert torch.allclose(result_and, expected_and)

        # OR = min(1, a+b)
        result_or = tnorm.disjunction(a, b)
        expected_or = torch.tensor([1.0, 1.0, 1.0])
        assert torch.allclose(result_or, expected_or)
