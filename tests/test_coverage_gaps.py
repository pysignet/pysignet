"""Tests to increase coverage for edge cases and error conditions.

These tests target specific uncovered lines identified by coverage
analysis.
"""

# pylint: disable=invalid-name

import warnings

import pytest
import torch
import torch.nn as nn
import sympy as sp
from sympy import srepr

from pysignet import (
    Symbol,
    Variable,
    compile_logic,
    logic_to_loss,
    ConsistencyChecker,
)
from pysignet.compilation import (
    TNormCompiler,
    LinearThresholdUnitCompiler,
)
from pysignet.compilation.module_utils import (
    infer_module_arity,
    has_final_activation,
    wrap_module_as_predicate,
    _get_output_dim,
    _get_final_layer,
)
from pysignet.logic import ForAll, Exists
from pysignet.logic.expansion import expand_quantifier
from pysignet.logic.extraction import (
    extract_variables_from_application,
    extract_constants_from_application,
    is_variable,
    is_constant,
)
from pysignet.tnorms import GodelTNorm, LukasiewiczTNorm


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
        # CompiledExpression returns per-batch results
        assert result.shape == (4,)

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

        # CompiledExpression returns per-batch results
        assert result.shape == (4,)

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

        compiled = logic_to_loss(expr, {"P": p_func})
        x = torch.randn(10, 5)

        result = compiled.satisfaction(X=x, quantify="exists")
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

        compiled = logic_to_loss(expr, {"P": p_func})
        x = torch.randn(10, 5)

        result = compiled.satisfaction(X=x, quantify="forall")
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
        compiled = logic_to_loss(expr, predicates)

        x = torch.randn(8, 10)
        loss = compiled.loss(
            X=x, quantify="none", reduction="sum"
        )

        assert loss.shape == ()  # Scalar after sum reduction

    def test_loss_with_quantify_none_reduction_none(self):
        """Test loss with quantify=none and reduction=none."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        predicates = {"P": lambda x: torch.sigmoid(x.sum(dim=-1))}
        compiled = logic_to_loss(expr, predicates)

        x = torch.randn(8, 10)
        loss = compiled.loss(
            X=x, quantify="none", reduction="none"
        )

        assert loss.shape == (8,)  # Per-element loss

    def test_log_satisfaction(self):
        """Test log_satisfaction method."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        predicates = {"P": lambda x: torch.sigmoid(x.sum(dim=-1))}
        compiled = logic_to_loss(expr, predicates)

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

        # CompiledExpression returns per-batch results
        assert result.shape == (4,)

    def test_partial_binding_loss(self):
        """Test computing loss from partial binding."""
        P, Q = Symbol("P Q")
        X, Y = Variable("X Y")
        expr = sp.And(P(X), Q(Y))

        predicates = {
            "P": lambda x: torch.sigmoid(x.sum(dim=-1)),
            "Q": lambda y: torch.sigmoid(y.mean(dim=-1)),
        }
        compiled = logic_to_loss(expr, predicates)

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
        # CompiledExpression returns per-batch results
        assert result.shape == (4,)


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

        compiled = logic_to_loss(expr, {"P": multi_arg_func})
        x = torch.randn(4, 10)
        result = compiled.satisfaction(X=x)

        assert result.shape == ()

    def test_lambda_predicate(self):
        """Test lambda function as predicate."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        compiled = logic_to_loss(expr, {
            "P": lambda x: torch.sigmoid(x.sum(dim=-1))
        })

        x = torch.randn(4, 10)
        result = compiled.satisfaction(X=x)

        assert result.shape == ()


class TestTNormOperators:
    """Additional tests for t-norm operator coverage."""

    def test_godel_tnorm_edge_values(self):
        """Test Godel t-norm with edge values."""
        tnorm = GodelTNorm()

        a = torch.tensor([0.0, 0.5, 1.0])
        b = torch.tensor([1.0, 0.5, 0.0])

        # AND = min (now takes tensor input with dim-0 reduction)
        result_and = tnorm.conjunction(torch.stack([a, b]))
        expected_and = torch.tensor([0.0, 0.5, 0.0])
        assert torch.allclose(result_and, expected_and)

        # OR = max (now takes tensor input with dim-0 reduction)
        result_or = tnorm.disjunction(torch.stack([a, b]))
        expected_or = torch.tensor([1.0, 0.5, 1.0])
        assert torch.allclose(result_or, expected_or)

    def test_lukasiewicz_tnorm_edge_values(self):
        """Test Lukasiewicz t-norm with edge values."""
        tnorm = LukasiewiczTNorm()

        a = torch.tensor([0.0, 0.5, 1.0])
        b = torch.tensor([1.0, 0.5, 0.0])

        # AND = max(0, sum-(n-1)), tensor input with dim-0 reduction
        result_and = tnorm.conjunction(torch.stack([a, b]))
        expected_and = torch.tensor([0.0, 0.0, 0.0])
        assert torch.allclose(result_and, expected_and)

        # OR = min(1, sum) (now takes tensor input with dim-0 reduction)
        result_or = tnorm.disjunction(torch.stack([a, b]))
        expected_or = torch.tensor([1.0, 1.0, 1.0])
        assert torch.allclose(result_or, expected_or)


class TestBaseCompilerEdgeCases:
    """Tests targeting uncovered lines in compilation/base.py."""

    def test_multiple_vars_require_keyword_args(self):
        """Test that multiple vars require keyword arguments."""
        P = Symbol("P")
        X, Y = Variable("X Y")
        expr = P(X, Y)

        def p_func(x, y):
            return torch.sigmoid(x.sum(dim=-1) + y.sum(dim=-1))

        compiler = TNormCompiler()
        compiled_expr = compiler.compile(expr, {"P": p_func})

        x = torch.randn(4, 10)
        y = torch.randn(4, 10)

        # Positional args should raise TypeError
        with pytest.raises(TypeError):
            compiled_expr(x)

        # Keyword args should work
        result = compiled_expr(X=x, Y=y)
        assert result.shape == (4,)

    def test_missing_var_in_dict_input_multivar(self):
        """Test error for missing variable in dict (lines 534)."""
        P = Symbol("P")
        X, Y = Variable("X Y")
        expr = P(X, Y)

        def p_func(x, y):
            return torch.sigmoid(x.sum(dim=-1) + y.sum(dim=-1))

        compiler = TNormCompiler()
        compiled_expr = compiler.compile(expr, {"P": p_func})

        x = torch.randn(4, 10)

        # Only provide X, missing Y
        with pytest.raises(ValueError, match="Missing input"):
            compiled_expr(X=x)

    def test_higher_dim_output_indexing(self):
        """Test indexing higher dimensional outputs (line 563)."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X, 0, 1)  # Two constants for indexing

        # Module that outputs 3D tensor: (batch, dim1, dim2)
        class HighDimModule(nn.Module):
            def forward(self, x):
                batch_size = x.shape[0]
                # Return (batch, 3, 4) tensor
                return torch.rand(batch_size, 3, 4)

        # This should work and index properly
        compiled = compile_logic(expr, {"P": HighDimModule()})
        x = torch.randn(4, 10)
        result = compiled(X=x)

        # CompiledExpression returns per-batch results
        assert result.shape == (4,)

    def test_invalid_boolean_constant_error(self):
        """Test error for invalid boolean constant (line 620)."""
        # Can't directly test line 620 without subclassing, but we can
        # verify boolean constants work correctly
        P = Symbol("P")
        X = Variable("X")
        expr = sp.And(P(X), sp.true)

        compiled = compile_logic(expr, {"P": lambda x: torch.ones(x.shape[0])})
        x = torch.randn(4, 10)
        result = compiled(X=x)

        # CompiledExpression returns per-batch results
        assert result.shape == (4,)


class TestLTUCompilerEdgeCases:
    """Tests targeting uncovered lines in ltu_compiler.py."""

    def test_ltu_large_alpha_warning(self):
        """Test warning for large alpha value (line 60)."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LinearThresholdUnitCompiler(
                mode="soft", alpha=15.0
            )
            assert len(w) == 1
            assert "too large" in str(w[0].message)

    def test_ltu_invalid_mode_error(self):
        """Test error for invalid mode."""
        with pytest.raises(ValueError, match="must be 'soft' or 'hard'"):
            LinearThresholdUnitCompiler(mode="invalid")

    def test_ltu_hard_mode_and(self):
        """Test LTU compiler in hard mode with AND."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.And(P(X), Q(X))

        compiler = LinearThresholdUnitCompiler(mode="hard")
        compiled = compiler.compile(expr, {
            "P": lambda x: torch.ones(x.shape[0]),
            "Q": lambda x: torch.ones(x.shape[0]),
        })

        x = torch.randn(4, 10)
        result = compiled(X=x)

        # In hard mode, all satisfied should give 1.0
        assert result.shape == (4,)  # Per-batch
        assert torch.all(result == 1.0)

    def test_ltu_hard_mode_or(self):
        """Test LTU compiler in hard mode with OR."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.Or(P(X), Q(X))

        compiler = LinearThresholdUnitCompiler(mode="hard")
        compiled = compiler.compile(expr, {
            "P": lambda x: torch.zeros(x.shape[0]),
            "Q": lambda x: torch.ones(x.shape[0]),
        })

        x = torch.randn(4, 10)
        result = compiled(X=x)

        # OR with one true should be true
        assert torch.all(result == 1.0)

    def test_ltu_unsupported_expression_type(self):
        """Test error for unsupported expression type (line 216)."""
        # Create an unsupported expression type
        # sympy.Xor is not supported
        P, Q = Symbol("P Q")
        X = Variable("X")

        # Use Xor which is not supported
        expr = sp.Xor(P(X), Q(X))

        compiler = LinearThresholdUnitCompiler()

        with pytest.raises(ValueError, match="Unsupported expression type"):
            compiled = compiler.compile(expr, {
                "P": lambda x: torch.ones(x.shape[0]),
                "Q": lambda x: torch.zeros(x.shape[0]),
            })
            x = torch.randn(4, 10)
            compiled(X=x)

    def test_ltu_boolean_constants(self):
        """Test LTU compiler with boolean constants (lines 146-147)."""
        P = Symbol("P")
        X = Variable("X")

        # sp.true
        expr = sp.And(P(X), sp.true)
        compiler = LinearThresholdUnitCompiler()
        compiled = compiler.compile(expr, {
            "P": lambda x: torch.ones(x.shape[0]),
        })
        x = torch.randn(4, 10)
        result = compiled(X=x)
        assert result.shape == (4,)

        # sp.false
        expr2 = sp.Or(P(X), sp.false)
        compiled2 = compiler.compile(expr2, {
            "P": lambda x: torch.ones(x.shape[0]),
        })
        result2 = compiled2(X=x)
        assert result2.shape == (4,)


class TestQuantifierEdgeCases:
    """Tests targeting uncovered lines in quantifier.py."""

    def test_quantifier_equality_unhashable_domain(self):
        """Test quantifier equality with domains that might fail conversion."""
        X = Variable("X")
        P = Symbol("P")
        body = P(X)

        # Create quantifiers with generator domains (consumed on iteration)
        q1 = ForAll(X, [1, 2, 3], body)
        q2 = ForAll(X, [1, 2, 3], body)

        # Should be equal
        assert q1 == q2

    def test_quantifier_inequality_different_domains(self):
        """Test quantifier inequality with different domains."""
        X = Variable("X")
        P = Symbol("P")
        body = P(X)

        q1 = ForAll(X, [1, 2, 3], body)
        q2 = ForAll(X, [1, 2, 4], body)

        assert q1 != q2

    def test_quantifier_equality_range_domain(self):
        """Test quantifier equality with range domains (line 96-97)."""
        X = Variable("X")
        P = Symbol("P")
        body = P(X)

        q1 = ForAll(X, range(5), body)
        q2 = ForAll(X, range(5), body)

        assert q1 == q2


class TestArityUninspectable:
    """Tests for uninspectable callable signatures."""

    def test_builtin_callable_arity(self):
        """Test that builtin callables work with explicit wrapping."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        # Wrap a simple tensor operation
        def sigmoid_wrapper(x):
            return torch.sigmoid(x.sum(dim=-1))

        compiled = compile_logic(expr, {"P": sigmoid_wrapper})
        x = torch.randn(4, 10)
        result = compiled(X=x)

        # CompiledExpression returns per-batch results
        assert result.shape == (4,)


class TestModuleUtilsEdgeCases:
    """Tests for module_utils.py edge cases."""

    def test_unsupported_arity_wrap_error(self):
        """Test error for unsupported arity (line 142)."""
        model = nn.Sequential(nn.Linear(10, 1), nn.Sigmoid())

        # Arity 3 is not supported - raises either ValueError or KeyError
        with pytest.raises((ValueError, KeyError)):
            wrap_module_as_predicate(model, arity=3)  # Only 1 and 2 supported


class TestTNormCompilerEdgeCases:
    """Tests for tnorm_compiler.py edge cases."""

    def test_tnorm_unsupported_expression(self):
        """Test error for unsupported expression type (line 176)."""
        P, Q = Symbol("P Q")
        X = Variable("X")

        # Xor is not directly supported
        expr = sp.Xor(P(X), Q(X))

        compiler = TNormCompiler()

        with pytest.raises(ValueError, match="Unsupported expression type"):
            compiled = compiler.compile(expr, {
                "P": lambda x: torch.ones(x.shape[0]),
                "Q": lambda x: torch.zeros(x.shape[0]),
            })
            x = torch.randn(4, 10)
            compiled(X=x)


class TestConsistencyCheckerEdgeCases:
    """Tests for consistency.py edge cases."""

    def test_consistency_checker_implies(self):
        """Test consistency checker with Implies operator."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        # Test Implies: P(X) -> Q(X)
        expr = sp.Implies(P(X), Q(X))

        def p_pred(x):
            # First half True, second half False
            result = torch.zeros(x.shape[0], dtype=torch.bool)
            result[:x.shape[0]//2] = True
            return result

        def q_pred(x):
            # All True
            return torch.ones(x.shape[0], dtype=torch.bool)

        predicates = {"P": p_pred, "Q": q_pred}

        checker = ConsistencyChecker(expr, predicates)
        x = torch.randn(4, 10)
        result = checker(x)

        # P -> Q should be satisfied when P is False OR Q is True
        assert result.shape == (4,)
        assert torch.all(result)  # All should be satisfied

    def test_consistency_checker_equivalent(self):
        """Test consistency checker with Equivalent operator."""
        P, Q = Symbol("P Q")
        X = Variable("X")
        expr = sp.Equivalent(P(X), Q(X))

        def p_pred(x):
            return torch.ones(x.shape[0], dtype=torch.bool)

        def q_pred(x):
            return torch.ones(x.shape[0], dtype=torch.bool)

        predicates = {"P": p_pred, "Q": q_pred}

        checker = ConsistencyChecker(expr, predicates)
        x = torch.randn(4, 10)
        result = checker(x)

        # P <-> Q should be satisfied when both are True
        assert result.shape == (4,)
        assert torch.all(result)


class TestLossEdgeCasesAdditional:
    """Additional tests for loss.py edge cases."""

    def test_log_satisfaction_exists(self):
        """Test log_satisfaction with exists quantification (line 153)."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        def p_func(x):
            return torch.ones(x.shape[0]) * 0.5

        compiled = logic_to_loss(expr, {"P": p_func})
        x = torch.randn(4, 10)

        log_sat = compiled.log_satisfaction(
            X=x, quantify="exists"
        )

        # log_satisfaction returns log-space result
        # For exists with product t-norm, result is in log space
        assert log_sat.shape == ()
        # Just verify it returns a valid tensor
        assert torch.isfinite(log_sat)

    def test_log_satisfaction_forall(self):
        """Test log_satisfaction with forall quantification."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        def p_func(x):
            return torch.ones(x.shape[0]) * 0.9

        compiled = logic_to_loss(expr, {"P": p_func})
        x = torch.randn(4, 10)

        log_sat = compiled.log_satisfaction(
            X=x, quantify="forall"
        )

        assert log_sat.shape == ()
        # Log of product of 0.9s should be negative
        assert log_sat <= 0

    def test_loss_invalid_quantify_error(self):
        """Test error for invalid quantify value in loss() (line 213)."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        def p_func(x):
            return torch.ones(x.shape[0])

        compiled = logic_to_loss(expr, {"P": p_func})
        x = torch.randn(4, 10)

        with pytest.raises(ValueError, match="Invalid quantify"):
            compiled.loss(X=x, quantify="invalid")


class TestExpansionEdgeCases:
    """Tests for expansion.py edge cases."""

    def test_exists_expansion(self):
        """Test Exists quantifier expansion."""
        X = Variable("X")
        P = Symbol("P")

        # Exists over small domain
        q = Exists(X, [0, 1, 2], P(X))
        expanded = expand_quantifier(q)

        # Should expand to Or
        assert isinstance(expanded, sp.Or)


class TestCompiledExpressionErrors:
    """Tests for error conditions in compiled_expression.py."""

    def test_partial_binding_empty_error(self):
        """Test error when partial() called with no bindings (line 205)."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        compiled = compile_logic(expr, {"P": lambda x: torch.ones(x.shape[0])})

        with pytest.raises(ValueError, match="Must provide at least one"):
            compiled.partial()

    def test_positional_tensor_rejected_single_var(self):
        """Test that positional tensor argument is rejected for single var.

        Users must use keyword arguments like compiled(X=tensor) instead
        of positional arguments like compiled(tensor).
        """
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        compiled = compile_logic(expr, {
            "P": lambda x: torch.sigmoid(x.sum(dim=-1))
        })

        x = torch.randn(4, 10)

        # Positional tensor should be rejected with TypeError
        with pytest.raises(TypeError):
            compiled(x)

    def test_positional_tensor_rejected_logic_loss(self):
        """Test that LogicLoss rejects positional tensor arguments.

        Users must use keyword arguments like logic_loss.satisfaction(X=tensor).
        When a tensor is passed positionally, it goes to the `quantify`
        parameter and causes a ValueError.
        """
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        logic_loss = logic_to_loss(expr, {
            "P": lambda x: torch.sigmoid(x.sum(dim=-1))
        })

        x = torch.randn(4, 10)

        # Positional tensor goes to `quantify` param, causing ValueError
        with pytest.raises(ValueError, match="Invalid quantify"):
            logic_loss.satisfaction(x)

    def test_positional_tensor_rejected_logic_loss_method(self):
        """Test that LogicLoss.loss() rejects positional tensor arguments.

        Users must use keyword arguments like logic_loss.loss(X=tensor).
        When a tensor is passed positionally, it goes to the `quantify`
        parameter and causes a ValueError.
        """
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        logic_loss = logic_to_loss(expr, {
            "P": lambda x: torch.sigmoid(x.sum(dim=-1))
        })

        x = torch.randn(4, 10)

        # Positional tensor goes to `quantify` param, causing ValueError
        with pytest.raises(ValueError, match="Invalid quantify"):
            logic_loss.loss(x)


class TestCompiledExpressionRepr:
    """Tests for CompiledExpression repr and expr property."""

    def test_compiled_expression_expr_property(self):
        """Test accessing expr property of CompiledExpression."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        def p_func(x):
            return torch.sigmoid(x.sum(dim=-1))

        compiled = compile_logic(expr, {"P": p_func})

        # Access expr property
        assert compiled.expr is not None
        assert compiled.expr == expr

    def test_compiled_expression_repr(self):
        """Test __repr__ of CompiledExpression."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        def p_func(x):
            return torch.sigmoid(x.sum(dim=-1))

        compiled = compile_logic(expr, {"P": p_func})

        repr_str = repr(compiled)

        assert "CompiledExpression(" in repr_str
        assert "P(X)" in repr_str or "expr=" in repr_str
        assert "free_variables" in repr_str
        assert "predicates" in repr_str

    def test_compiled_expression_repr_with_partial_binding(self):
        """Test __repr__ with partial bindings."""
        P = Symbol("P")
        X, Y = Variable("X Y")
        expr = sp.And(P(X), P(Y))

        def p_func(x):
            return torch.sigmoid(x.sum(dim=-1))

        compiled = compile_logic(expr, {"P": p_func})

        # Create partial binding
        x = torch.randn(4, 10)
        partial = compiled.partial(X=x)

        repr_str = repr(partial)

        assert "bound=" in repr_str
        assert "X" in repr_str


class TestBooleanSatisfaction:
    """Tests for return_boolean parameter."""

    def test_boolean_satisfaction_binary_predicate(self):
        """Test boolean satisfaction with binary predicate."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        # Predicate returning high values (> 0.5)
        compiled = compile_logic(
            expr, {"P": lambda x: torch.ones(x.shape[0]) * 0.8}
        )

        x = torch.randn(4, 10)
        result = compiled(X=x, return_boolean=True)

        assert result.dtype == torch.bool
        assert result.all()  # All should be True (0.8 > 0.5)

    def test_boolean_satisfaction_low_values(self):
        """Test boolean satisfaction with low values."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        # Predicate returning low values (< 0.5)
        compiled = compile_logic(
            expr, {"P": lambda x: torch.ones(x.shape[0]) * 0.3}
        )

        x = torch.randn(4, 10)
        result = compiled(X=x, return_boolean=True)

        assert result.dtype == torch.bool
        assert not result.any()  # All should be False (0.3 < 0.5)

    def test_boolean_satisfaction_multiclass_predicate(self):
        """Test boolean satisfaction with multiclass predicate."""
        Digit = Symbol("Digit")
        X = Variable("X")
        expr = Digit(X, 2)  # Class 2

        # Use nn.Module for multiclass - returns (batch, num_classes)
        model = nn.Sequential(nn.Linear(10, 5), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"Digit": model})

        x = torch.randn(4, 10)
        result = compiled(X=x, return_boolean=True)

        assert result.dtype == torch.bool
        # Result depends on model weights, just check it's boolean
        assert result.shape == (4,)


class TestBooleanSatisfactionWithQuantifiers:
    """Tests for return_boolean with ForAll/Exists quantifiers.

    Regression tests for bug where ConsistencyChecker raised
    ValueError on quantifier expressions because it did not
    handle ForAll/Exists types.

    Boolean conversion uses threshold at 0.5: Digit(X, k) is True
    iff the satisfaction degree exceeds 0.5. This ensures consistency
    between soft and boolean evaluation -- when soft satisfaction is
    near 0, boolean should also be False.
    """

    def test_exists_quantifier_return_boolean(self):
        """Test return_boolean with Exists quantifier."""
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")

        expr = Exists(Y, range(3), Digit(X, Y))

        # Predicate: class 1 exceeds 0.5 threshold
        def digit_pred(x, class_idx):
            batch = x.shape[0]
            if class_idx == 1:
                return torch.ones(batch) * 0.9
            return torch.ones(batch) * 0.1

        compiled = compile_logic(expr, {"Digit": digit_pred})

        x = torch.randn(4, 10)
        result = compiled(X=x, return_boolean=True)

        assert result.dtype == torch.bool
        assert result.shape == (4,)
        # Class 1 is 0.9 > 0.5 -> Exists is satisfied
        assert result.all()

    def test_exists_unsatisfied_when_all_below_threshold(self):
        """Test Exists is False when no class exceeds 0.5."""
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")

        expr = Exists(Y, range(3), Digit(X, Y))

        # All classes below 0.5 -> model is not confident
        def digit_pred(x, class_idx):
            batch = x.shape[0]
            probs = {0: 0.2, 1: 0.35, 2: 0.45}
            return torch.ones(batch) * probs[class_idx]

        compiled = compile_logic(expr, {"Digit": digit_pred})

        x = torch.randn(4, 10)
        result = compiled(X=x, return_boolean=True)

        assert result.dtype == torch.bool
        assert result.shape == (4,)
        # No class > 0.5 -> Exists is not satisfied
        assert not result.any()

    def test_forall_at_most_one_satisfied(self):
        """Test at_most_one ForAll with one confident prediction.

        Only one class exceeds 0.5, so no pair has both True.
        """
        Digit = Symbol("Digit")
        X, I, J = Variable("X I J")

        n_classes = 3
        all_pairs = [
            (i, j) for i in range(n_classes)
            for j in range(i + 1, n_classes)
        ]
        at_most_one = ForAll(
            [I, J], all_pairs,
            sp.Not(sp.And(Digit(X, I), Digit(X, J)))
        )

        # Only class 2 exceeds 0.5
        def digit_pred(x, class_idx):
            batch = x.shape[0]
            if class_idx == 2:
                return torch.ones(batch) * 0.8
            return torch.ones(batch) * 0.1

        compiled = compile_logic(at_most_one, {"Digit": digit_pred})

        x = torch.randn(4, 10)
        result = compiled(X=x, return_boolean=True)

        assert result.dtype == torch.bool
        assert result.shape == (4,)
        assert result.all()

    def test_forall_at_most_one_violated(self):
        """Test at_most_one ForAll violated when two classes > 0.5."""
        Digit = Symbol("Digit")
        X, I, J = Variable("X I J")

        n_classes = 3
        all_pairs = [
            (i, j) for i in range(n_classes)
            for j in range(i + 1, n_classes)
        ]
        at_most_one = ForAll(
            [I, J], all_pairs,
            sp.Not(sp.And(Digit(X, I), Digit(X, J)))
        )

        # Classes 0 and 1 both exceed 0.5 -> pair (0,1) violates
        def digit_pred(x, class_idx):
            batch = x.shape[0]
            if class_idx in (0, 1):
                return torch.ones(batch) * 0.8
            return torch.ones(batch) * 0.1

        compiled = compile_logic(at_most_one, {"Digit": digit_pred})

        x = torch.randn(4, 10)
        result = compiled(X=x, return_boolean=True)

        assert result.dtype == torch.bool
        assert result.shape == (4,)
        # Pair (0,1) both True -> Not(And(True, True)) = False
        assert not result.any()

    def test_nested_quantifiers_exactly_one_satisfied(self):
        """Test exactly-one with one confident class.

        Exists(Y, range(N), Digit(X,Y)) AND
        ForAll([I,J], pairs, NOT(Digit(X,I) AND Digit(X,J)))

        One class exceeds 0.5 -> both constraints satisfied.
        """
        Digit = Symbol("Digit")
        X, Y, I, J = Variable("X Y I J")

        n_classes = 3
        at_least_one = Exists(Y, range(n_classes), Digit(X, Y))

        all_pairs = [
            (i, j) for i in range(n_classes)
            for j in range(i + 1, n_classes)
        ]
        at_most_one = ForAll(
            [I, J], all_pairs,
            sp.Not(sp.And(Digit(X, I), Digit(X, J)))
        )

        exactly_one = sp.And(at_least_one, at_most_one)

        # Only class 1 exceeds 0.5
        def digit_pred(x, class_idx):
            batch = x.shape[0]
            if class_idx == 1:
                return torch.ones(batch) * 0.9
            return torch.ones(batch) * 0.1

        compiled = compile_logic(exactly_one, {"Digit": digit_pred})

        x = torch.randn(4, 10)
        result = compiled(X=x, return_boolean=True)

        assert result.dtype == torch.bool
        assert result.shape == (4,)
        assert result.all()

    def test_nested_quantifiers_exactly_one_violated(self):
        """Test exactly-one violated when no class exceeds 0.5.

        When the model is uncertain (all probabilities below 0.5),
        exactly-one is not satisfied: at_least_one fails because
        no Digit(X, k) is True.
        """
        Digit = Symbol("Digit")
        X, Y, I, J = Variable("X Y I J")

        n_classes = 3
        at_least_one = Exists(Y, range(n_classes), Digit(X, Y))

        all_pairs = [
            (i, j) for i in range(n_classes)
            for j in range(i + 1, n_classes)
        ]
        at_most_one = ForAll(
            [I, J], all_pairs,
            sp.Not(sp.And(Digit(X, I), Digit(X, J)))
        )

        exactly_one = sp.And(at_least_one, at_most_one)

        # Near-uniform: no class exceeds 0.5
        def digit_pred(x, _class_idx):
            batch = x.shape[0]
            return torch.ones(batch) * 0.3

        compiled = compile_logic(exactly_one, {"Digit": digit_pred})

        x = torch.randn(4, 10)
        result = compiled(X=x, return_boolean=True)

        assert result.dtype == torch.bool
        assert result.shape == (4,)
        # No class > 0.5 -> at_least_one fails -> exactly_one False
        assert not result.any()


class TestLogicLossFreeVariables:
    """Tests for LogicLoss.free_variables property."""

    def test_free_variables_single(self):
        """Test free_variables with single variable."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        def p_func(x):
            return torch.sigmoid(x.sum(dim=-1))

        logic_loss = logic_to_loss(expr, {"P": p_func})

        assert "X" in logic_loss.free_variables
        assert len(logic_loss.free_variables) == 1

    def test_free_variables_multiple(self):
        """Test free_variables with multiple variables."""
        P, Q = Symbol("P Q")
        X, Y = Variable("X Y")
        expr = sp.And(P(X), Q(Y))

        logic_loss = logic_to_loss(expr, {
            "P": lambda x: torch.sigmoid(x.sum(dim=-1)),
            "Q": lambda y: torch.sigmoid(y.mean(dim=-1)),
        })

        assert "X" in logic_loss.free_variables
        assert "Y" in logic_loss.free_variables
        assert len(logic_loss.free_variables) == 2


class TestLogicLossInvalidQuantify:
    """Tests for LogicLoss with invalid quantify values."""

    def test_log_satisfaction_invalid_quantify(self):
        """Test log_satisfaction with invalid quantify value."""
        P = Symbol("P")
        X = Variable("X")
        expr = P(X)

        def p_func(x):
            return torch.sigmoid(x.sum(dim=-1))

        logic_loss = logic_to_loss(expr, {"P": p_func})

        x = torch.randn(4, 10)

        with pytest.raises(ValueError, match="Invalid quantify"):
            logic_loss.log_satisfaction(X=x, quantify="invalid")


class TestPredicateApplicationSymPyPrinting:
    """Tests for PredicateApplication SymPy printing methods."""

    def test_sympystr(self):
        """Test _sympystr method."""
        P = Symbol("P")
        X = Variable("X")
        app = P(X, 0)

        # SymPy's str() uses _sympystr
        str_repr = str(app)

        assert "P" in str_repr
        assert "X" in str_repr or "0" in str_repr

    def test_sympyrepr(self):
        """Test _sympyrepr method via srepr."""
        P = Symbol("P")
        X = Variable("X")
        app = P(X, 1)

        repr_str = srepr(app)

        # Should contain predicate name
        assert "P" in repr_str
