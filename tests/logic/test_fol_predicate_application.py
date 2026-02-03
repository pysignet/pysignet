"""Tests for predicate application with FOL variables.

This module tests that PredicateApplication can handle variables
in addition to concrete arguments.
"""

import pytest
import sympy as sp

from pysignet import Symbol
from pysignet.logic import Variable
from pysignet.symbols import PredicateApplication


class TestPredicateApplicationWithVariables:
    """Test predicate application with variable arguments."""

    def test_unary_predicate_with_variable(self):
        """Test applying unary predicate to a variable."""
        Digit = Symbol("Digit")
        X = Variable("X")

        # Apply predicate to variable
        app = Digit(X)

        # Should be a PredicateApplication
        assert isinstance(app, PredicateApplication)

        # Should have correct predicate name
        assert app.predicate_name == "Digit"

        # Should have variable as argument
        assert app.application_args == (X,)
        assert len(app.application_args) == 1

    def test_binary_predicate_with_two_variables(self):
        """Test applying binary predicate to two variables."""
        Rel = Symbol("Rel")
        X, Y = Variable("X Y")

        # Apply predicate to two variables
        app = Rel(X, Y)

        assert isinstance(app, PredicateApplication)
        assert app.predicate_name == "Rel"
        assert app.application_args == (X, Y)
        assert len(app.application_args) == 2

    def test_ternary_predicate_with_three_variables(self):
        """Test applying ternary predicate to three variables."""
        Q = Symbol("Q")
        X, Y, Z = Variable("X Y Z")

        # Apply predicate to three variables
        app = Q(X, Y, Z)

        assert isinstance(app, PredicateApplication)
        assert app.predicate_name == "Q"
        assert app.application_args == (X, Y, Z)
        assert len(app.application_args) == 3

    def test_mixed_variable_and_constant_args(self):
        """Test predicate with both variable and constant arguments."""
        Digit = Symbol("Digit")
        X = Variable("X")

        # Digit(X, 5) - X is variable, 5 is constant
        app = Digit(X, 5)

        assert isinstance(app, PredicateApplication)
        assert app.predicate_name == "Digit"
        assert len(app.application_args) == 2
        assert app.application_args[0] == X
        assert app.application_args[1] == 5

    def test_mixed_constant_and_variable_args(self):
        """Test predicate with constant then variable."""
        P = Symbol("P")
        Y = Variable("Y")

        # P(3, Y) - 3 is constant, Y is variable
        app = P(3, Y)

        assert isinstance(app, PredicateApplication)
        assert app.predicate_name == "P"
        assert app.application_args == (3, Y)

    def test_multiple_mixed_args(self):
        """Test predicate with multiple mixed args."""
        Q = Symbol("Q")
        X, Z = Variable("X Z")

        # Q(X, 1, Z, 4) - alternating
        app = Q(X, 1, Z, 4)

        assert isinstance(app, PredicateApplication)
        assert app.predicate_name == "Q"
        assert app.application_args == (X, 1, Z, 4)
        assert len(app.application_args) == 4


class TestPredicateApplicationEquality:
    """Test equality of predicate applications with variables."""

    def test_same_variable_applications_are_equal(self):
        """Test that applications with same variable are equal."""
        Digit = Symbol("Digit")
        X = Variable("X")

        app1 = Digit(X)
        app2 = Digit(X)

        assert app1 == app2

    def test_different_variable_applications_not_equal(self):
        """Test that applications with different variables are not equal."""
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")

        app1 = Digit(X)
        app2 = Digit(Y)

        assert app1 != app2

    def test_variable_vs_constant_not_equal(self):
        """Test that applications with different args are not equal."""
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")

        app_x = Digit(X)
        app_y = Digit(Y)

        assert app_x != app_y

    def test_mixed_args_equality(self):
        """Test equality with mixed arguments."""
        P = Symbol("P")
        X, Y = Variable("X Y")

        app1 = P(X, 5)
        app2 = P(X, 5)
        app3 = P(Y, 5)
        app4 = P(X, 6)

        assert app1 == app2
        assert app1 != app3  # Different variable
        assert app1 != app4  # Different constant


class TestPredicateApplicationHashing:
    """Test hashing of predicate applications with variables."""

    def test_variable_application_hashable(self):
        """Test that applications with variables are hashable."""
        Digit = Symbol("Digit")
        X, Y = Variable("X Y")

        app1 = Digit(X)
        app2 = Digit(Y)

        # Should be hashable
        app_set = {app1, app2, app1}  # app1 appears twice
        assert len(app_set) == 2  # Should deduplicate

        # Should work as dict keys
        app_dict = {app1: 1, app2: 2}
        assert app_dict[app1] == 1
        assert app_dict[app2] == 2


class TestPredicateApplicationInExpressions:
    """Test using predicate applications with variables in logical expressions."""

    def test_variable_application_in_and(self):
        """Test variable application in AND expression."""
        P, Q = Symbol("P Q")
        X = Variable("X")

        expr = sp.And(P(X), Q(X))

        assert isinstance(expr, sp.And)
        assert len(expr.args) == 2

    def test_variable_application_in_or(self):
        """Test variable application in OR expression."""
        Digit = Symbol("Digit")
        X = Variable("X")

        expr = sp.Or(Digit(X, 0), Digit(X, 1), Digit(X, 2))

        assert isinstance(expr, sp.Or)
        assert len(expr.args) == 3

    def test_variable_application_in_implies(self):
        """Test variable application in IMPLIES expression."""
        P, Q = Symbol("P Q")
        X = Variable("X")

        expr = sp.Implies(P(X), Q(X))

        assert isinstance(expr, sp.Implies)

    def test_variable_application_in_not(self):
        """Test variable application in NOT expression."""
        P = Symbol("P")
        X = Variable("X")

        expr = sp.Not(P(X))

        assert isinstance(expr, sp.Not)

    def test_complex_expression_with_variables(self):
        """Test complex expression with multiple variables."""
        P, Q, R = Symbol("P Q R")
        X, Y = Variable("X Y")

        # (P(X) ∧ Q(X, Y)) → R(Y)
        expr = sp.Implies(
            sp.And(P(X), Q(X, Y)),
            R(Y)
        )

        assert isinstance(expr, sp.Implies)


class TestPredicateApplicationRepresentation:
    """Test string representation of predicate applications with variables."""

    def test_variable_repr(self):
        """Test repr of predicate with variable."""
        Digit = Symbol("Digit")
        X = Variable("X")

        app = Digit(X)

        # Should show variable name
        assert str(app) == "Digit(X)"
        assert repr(app) == "Digit(X)"

    def test_mixed_args_repr(self):
        """Test repr with mixed variable and constant args."""
        P = Symbol("P")
        X, Y = Variable("X Y")

        app = P(X, 5, Y)

        # Should show both variables and constants
        assert "X" in str(app)
        assert "5" in str(app)
        assert "Y" in str(app)


class TestPredicateApplicationEdgeCases:
    """Test edge cases for predicate applications with variables."""

    def test_same_variable_multiple_times(self):
        """Test using same variable multiple times in one application."""
        Rel = Symbol("Rel")
        X = Variable("X")

        # Rel(X, X) - same variable twice
        app = Rel(X, X)

        assert isinstance(app, PredicateApplication)
        assert app.application_args == (X, X)
        assert len(app.application_args) == 2

    def test_no_args_still_works(self):
        """Test that nullary (no args) predicates still work."""
        P, Q = Symbol("P Q")

        # P and Q are used without calling - should work as before
        expr = sp.And(P, Q)

        assert isinstance(expr, sp.And)

    def test_mixed_variable_and_constant(self):
        """Test that mixed variable and constant applications work."""
        Digit = Symbol("Digit")
        X = Variable("X")

        # FOL interface: variable X with constant 0
        app = Digit(X, 0)

        assert isinstance(app, PredicateApplication)
        assert len(app.application_args) == 2
