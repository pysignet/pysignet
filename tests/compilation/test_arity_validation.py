"""Tests for arity validation in compilation/arity.py.

This module tests the new clean arity validation implementation that:
1. Disallows nullary predicates (Symbol without arguments)
2. Validates PredicateApplication arity matches callable signature
3. Provides clear error messages
4. No special cases (nn.Module handled by module_utils wrapping)
"""

import pytest
import sympy as sp
import torch
import torch.nn as nn

from pysignet import Symbol, Variable
from pysignet.compilation.arity import validate_predicate_arity
from pysignet.predicate import Predicate


class TestPredicateApplicationArity:
    """Test arity validation for PredicateApplication nodes."""

    def test_unary_predicate_valid(self):
        """Test P(X) with lambda x: ... is valid."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {
            "P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1)))
        }

        # Should not raise
        validate_predicate_arity(expr, predicates)

    def test_binary_predicate_valid(self):
        """Test P(X, Y) with lambda x, y: ... is valid."""
        X, Y = Variable("X Y")
        P = Symbol("P")
        expr = P(X, Y)

        predicates = {
            "P": Predicate(lambda x, y: torch.sigmoid(x.sum(dim=-1) + y))
        }

        # Should not raise
        validate_predicate_arity(expr, predicates)

    def test_ternary_predicate_valid(self):
        """Test P(X, Y, Z) with lambda x, y, z: ... is valid."""
        X, Y, Z = Variable("X Y Z")
        P = Symbol("P")
        expr = P(X, Y, Z)

        predicates = {
            "P": Predicate(lambda x, y, z: torch.sigmoid(x.sum(dim=-1) + y + z))
        }

        # Should not raise
        validate_predicate_arity(expr, predicates)

    def test_mixed_variable_constant_valid(self):
        """Test P(X, 0, Y) with lambda x, c, y: ... is valid."""
        X, Y = Variable("X Y")
        P = Symbol("P")
        expr = P(X, 0, Y)

        predicates = {
            "P": Predicate(lambda x, c, y: torch.sigmoid(x[:, c] + y))
        }

        # Should not raise
        validate_predicate_arity(expr, predicates)

    def test_arity_mismatch_too_few_args(self):
        """Test P(X, Y) with lambda x: ... raises error."""
        X, Y = Variable("X Y")
        P = Symbol("P")
        expr = P(X, Y)

        predicates = {
            "P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1)))
        }

        with pytest.raises(ValueError) as exc_info:
            validate_predicate_arity(expr, predicates)

        error_msg = str(exc_info.value)
        assert "P" in error_msg
        assert "2 argument(s)" in error_msg
        assert "1 argument(s)" in error_msg
        assert "arity mismatch" in error_msg.lower()

    def test_arity_mismatch_too_many_args(self):
        """Test P(X) with lambda x, y: ... raises error."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        predicates = {
            "P": Predicate(lambda x, y: torch.sigmoid(x.sum(dim=-1) + y))
        }

        with pytest.raises(ValueError) as exc_info:
            validate_predicate_arity(expr, predicates)

        error_msg = str(exc_info.value)
        assert "P" in error_msg
        assert "1 argument(s)" in error_msg
        assert "2 argument(s)" in error_msg

    def test_multiple_predicates_all_valid(self):
        """Test expression with multiple predicates, all valid."""
        X = Variable("X")
        P, Q, R = Symbol("P Q R")
        expr = sp.And(P(X), sp.Or(Q(X), sp.Not(R(X))))

        predicates = {
            "P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1))),
            "Q": Predicate(lambda x: torch.sigmoid(x.mean(dim=-1))),
            "R": Predicate(lambda x: (x > 0).float().mean(dim=-1)),
        }

        # Should not raise
        validate_predicate_arity(expr, predicates)

    def test_multiple_predicates_one_invalid(self):
        """Test expression with multiple predicates, one invalid."""
        X, Y = Variable("X Y")
        P, Q = Symbol("P Q")
        expr = sp.And(P(X), Q(X, Y))

        predicates = {
            "P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1))),
            "Q": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1))),  # Wrong!
        }

        with pytest.raises(ValueError) as exc_info:
            validate_predicate_arity(expr, predicates)

        # Should complain about Q
        error_msg = str(exc_info.value)
        assert "Q" in error_msg


class TestNullaryPredicateDisallowed:
    """Test that nullary predicates (bare symbols) are disallowed."""

    def test_nullary_symbol_raises_error(self):
        """Test that bare symbol P raises error."""
        P = Symbol("P")
        expr = P  # No arguments - nullary

        predicates = {
            "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.9)
        }

        with pytest.raises(ValueError) as exc_info:
            validate_predicate_arity(expr, predicates)

        error_msg = str(exc_info.value)
        assert "nullary" in error_msg.lower() or "without arguments" in error_msg.lower()
        assert "P(X)" in error_msg  # Suggest correct usage

    def test_nullary_in_expression_raises_error(self):
        """Test that nullary in larger expression raises error."""
        X = Variable("X")
        P, Q = Symbol("P Q")
        expr = sp.And(P, Q(X))  # P is nullary

        predicates = {
            "P": Predicate(lambda x: torch.ones(x.shape[0]) * 0.9),
            "Q": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1))),
        }

        with pytest.raises(ValueError) as exc_info:
            validate_predicate_arity(expr, predicates)

        error_msg = str(exc_info.value)
        assert "P" in error_msg

    def test_sp_true_false_allowed(self):
        """Test that sp.true and sp.false are still allowed."""
        X = Variable("X")
        P = Symbol("P")
        expr = sp.And(sp.true, P(X))

        predicates = {
            "P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1)))
        }

        # Should not raise - sp.true/sp.false are not predicates
        validate_predicate_arity(expr, predicates)


class TestBoundMethods:
    """Test arity validation for bound methods."""

    def test_bound_method_excludes_self(self):
        """Test that 'self' parameter is excluded from arity count."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        class Model:
            def predict(self, x):
                return torch.sigmoid(x.sum(dim=-1))

        model = Model()
        predicates = {
            "P": Predicate(model.predict)
        }

        # Should not raise - bound method has 2 params (self, x) but self excluded
        validate_predicate_arity(expr, predicates)


class TestEdgeCases:
    """Test edge cases in arity validation."""

    def test_duplicate_variables_in_arguments(self):
        """Test P(X, X, 0) - duplicate variable."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X, X, 0)

        # Callable must accept 3 arguments (even though X appears twice)
        predicates = {
            "P": Predicate(lambda x1, x2, c: torch.sigmoid(x1.sum(dim=-1) + x2.sum(dim=-1)))
        }

        # Should not raise
        validate_predicate_arity(expr, predicates)

    def test_only_constants_valid(self):
        """Test P(0, 1, 2) with lambda a, b, c: ... is valid."""
        P = Symbol("P")
        expr = P(0, 1, 2)

        predicates = {
            "P": Predicate(lambda a, b, c: torch.tensor(0.5))
        }

        # Should not raise - valid arity (3 args)
        validate_predicate_arity(expr, predicates)

    def test_nested_expressions(self):
        """Test validation recursively walks nested expressions."""
        X, Y = Variable("X Y")
        P, Q, R = Symbol("P Q R")
        expr = sp.Implies(
            sp.And(P(X), Q(X, Y)),
            sp.Or(R(X), sp.Not(P(X)))
        )

        predicates = {
            "P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1))),
            "Q": Predicate(lambda x, y: torch.sigmoid(x.sum(dim=-1) + y)),
            "R": Predicate(lambda x: (x > 0).float().mean(dim=-1)),
        }

        # Should not raise
        validate_predicate_arity(expr, predicates)

    def test_predicate_not_in_dict_skipped(self):
        """Test that symbols not in predicates dict are skipped."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        # Q not in predicates - should be handled elsewhere (symbol extraction)
        predicates = {}

        # Should not raise - validation only checks predicates in dict
        # (Symbol extraction will catch missing predicates)
        validate_predicate_arity(expr, predicates)


class TestErrorMessages:
    """Test that error messages are helpful and clear."""

    def test_error_message_includes_predicate_name(self):
        """Test error message includes predicate name."""
        X, Y = Variable("X Y")
        MyPred = Symbol("MyPredicate")
        expr = MyPred(X, Y)

        predicates = {
            "MyPredicate": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1)))
        }

        with pytest.raises(ValueError) as exc_info:
            validate_predicate_arity(expr, predicates)

        assert "MyPredicate" in str(exc_info.value)

    def test_error_message_includes_expected_actual(self):
        """Test error message includes expected and actual arity."""
        X, Y, Z = Variable("X Y Z")
        P = Symbol("P")
        expr = P(X, Y, Z)

        predicates = {
            "P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1)))
        }

        with pytest.raises(ValueError) as exc_info:
            validate_predicate_arity(expr, predicates)

        error_msg = str(exc_info.value)
        assert "3" in error_msg  # Expected
        assert "1" in error_msg  # Actual

    def test_error_message_shows_usage(self):
        """Test error message shows the problematic usage."""
        X, Y = Variable("X Y")
        P = Symbol("P")
        expr = P(X, Y)

        predicates = {
            "P": Predicate(lambda x: torch.sigmoid(x.sum(dim=-1)))
        }

        with pytest.raises(ValueError) as exc_info:
            validate_predicate_arity(expr, predicates)

        error_msg = str(exc_info.value)
        # Should show the application like "P(X, Y)"
        assert "P" in error_msg
