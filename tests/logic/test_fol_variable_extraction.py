"""Tests for variable extraction from logical expressions.

This module tests the extract_variables() function which identifies all
free variables in a logical expression containing PredicateApplications.
"""

import pytest
import sympy as sp

from pysignet import Symbol
from pysignet.logic import Variable, extract_variables
from pysignet.multiclass import PredicateApplication


class TestBasicVariableExtraction:
    """Test basic variable extraction from simple expressions."""

    def test_extract_from_single_predicate_application(self):
        """Test extracting variable from single predicate application."""
        Digit = Symbol("Digit")
        X = Variable("X")

        expr = Digit(X)
        variables = extract_variables(expr)

        assert len(variables) == 1
        assert X in variables

    def test_extract_from_multiple_applications_same_variable(self):
        """Test that same variable is only extracted once."""
        P, Q = Symbol("P Q")
        X = Variable("X")

        # P(X) ∧ Q(X) - X appears twice but should be extracted once
        expr = sp.And(P(X), Q(X))
        variables = extract_variables(expr)

        assert len(variables) == 1
        assert X in variables

    def test_extract_from_multiple_different_variables(self):
        """Test extracting multiple different variables."""
        P, Q = Symbol("P Q")
        X, Y = Variable("X Y")

        # P(X) ∧ Q(Y)
        expr = sp.And(P(X), Q(Y))
        variables = extract_variables(expr)

        assert len(variables) == 2
        assert X in variables
        assert Y in variables

    def test_extract_from_binary_predicate(self):
        """Test extracting variables from binary predicate."""
        Rel = Symbol("Rel")
        X, Y = Variable("X Y")

        expr = Rel(X, Y)
        variables = extract_variables(expr)

        assert len(variables) == 2
        assert X in variables
        assert Y in variables

    def test_extract_from_mixed_args(self):
        """Test extracting variables from mixed variable/constant args."""
        Digit = Symbol("Digit")
        X = Variable("X")

        # Digit(X, 5) - only X should be extracted
        expr = Digit(X, 5)
        variables = extract_variables(expr)

        assert len(variables) == 1
        assert X in variables


class TestComplexExpressions:
    """Test variable extraction from complex logical expressions."""

    def test_extract_from_nested_and_or(self):
        """Test extraction from nested AND/OR expressions."""
        P, Q, R = Symbol("P Q R")
        X, Y, Z = Variable("X Y Z")

        # (P(X) ∧ Q(Y)) ∨ R(Z)
        expr = sp.Or(sp.And(P(X), Q(Y)), R(Z))
        variables = extract_variables(expr)

        assert len(variables) == 3
        assert X in variables
        assert Y in variables
        assert Z in variables

    def test_extract_from_implication(self):
        """Test extraction from implication."""
        P, Q = Symbol("P Q")
        X, Y = Variable("X Y")

        # P(X) → Q(Y)
        expr = sp.Implies(P(X), Q(Y))
        variables = extract_variables(expr)

        assert len(variables) == 2
        assert X in variables
        assert Y in variables

    def test_extract_from_negation(self):
        """Test extraction from negation."""
        P = Symbol("P")
        X = Variable("X")

        # ¬P(X)
        expr = sp.Not(P(X))
        variables = extract_variables(expr)

        assert len(variables) == 1
        assert X in variables

    def test_extract_from_equivalence(self):
        """Test extraction from equivalence."""
        P, Q = Symbol("P Q")
        X, Y = Variable("X Y")

        # P(X) ↔ Q(Y)
        expr = sp.Equivalent(P(X), Q(Y))
        variables = extract_variables(expr)

        assert len(variables) == 2
        assert X in variables
        assert Y in variables

    def test_extract_from_deeply_nested_expression(self):
        """Test extraction from deeply nested expression."""
        P, Q, R, S = Symbol("P Q R S")
        X, Y, Z = Variable("X Y Z")

        # ((P(X) ∧ Q(Y)) → R(Z)) ∨ S(X)
        expr = sp.Or(
            sp.Implies(sp.And(P(X), Q(Y)), R(Z)),
            S(X)
        )
        variables = extract_variables(expr)

        # X, Y, Z should all be present (X appears twice but counted once)
        assert len(variables) == 3
        assert X in variables
        assert Y in variables
        assert Z in variables


class TestEdgeCases:
    """Test edge cases for variable extraction."""

    def test_extract_from_constant_only_predicate(self):
        """Test extraction from predicate with only constant args."""
        R = Symbol("R")

        # R(0) - no variables, just constant
        expr = R(0)
        variables = extract_variables(expr)

        assert len(variables) == 0

    def test_extract_from_nullary_predicate(self):
        """Test extraction from nullary predicate (no args)."""
        P = Symbol("P")

        # P - no variables
        expr = P
        variables = extract_variables(expr)

        assert len(variables) == 0

    def test_extract_from_boolean_constant(self):
        """Test extraction from boolean constant."""
        # sp.true - no variables
        expr = sp.true
        variables = extract_variables(expr)

        assert len(variables) == 0

    def test_extract_from_expression_with_constants_and_variables(self):
        """Test extraction from expression mixing constants and variables."""
        P, Q = Symbol("P Q")
        X = Variable("X")

        # P ∧ Q(X) - P has no args, Q has variable
        expr = sp.And(P, Q(X))
        variables = extract_variables(expr)

        assert len(variables) == 1
        assert X in variables

    def test_extract_same_variable_in_multiple_positions(self):
        """Test extracting when same variable appears multiple times."""
        Rel = Symbol("Rel")
        X = Variable("X")

        # Rel(X, X) - same variable twice in one predicate
        expr = Rel(X, X)
        variables = extract_variables(expr)

        assert len(variables) == 1
        assert X in variables

    def test_extract_from_complex_mixed_expression(self):
        """Test extraction from complex expression with mixed args."""
        P, Q, R = Symbol("P Q R")
        X, Y = Variable("X Y")

        # P(X, 5) ∧ Q(Y, 3, X) ∧ R(7)
        expr = sp.And(P(X, 5), Q(Y, 3, X), R(7))
        variables = extract_variables(expr)

        # Only X and Y (constants 5, 3, 7 not extracted)
        assert len(variables) == 2
        assert X in variables
        assert Y in variables


class TestReturnType:
    """Test the return type and properties of extracted variables."""

    def test_returns_set(self):
        """Test that extract_variables returns a set."""
        Digit = Symbol("Digit")
        X = Variable("X")

        expr = Digit(X)
        variables = extract_variables(expr)

        assert isinstance(variables, set)

    def test_returns_empty_set_for_no_variables(self):
        """Test that empty set is returned when no variables."""
        P = Symbol("P")

        expr = P
        variables = extract_variables(expr)

        assert isinstance(variables, set)
        assert len(variables) == 0

    def test_set_contains_variable_symbols(self):
        """Test that set contains VariableSymbol instances."""
        from pysignet.logic.variable import VariableSymbol

        Digit = Symbol("Digit")
        X, Y = Variable("X Y")

        expr = sp.And(Digit(X), Digit(Y))
        variables = extract_variables(expr)

        for var in variables:
            assert isinstance(var, VariableSymbol)


class TestRecursiveTraversal:
    """Test that variable extraction traverses expression tree correctly."""

    def test_traverses_and_operator(self):
        """Test traversal through AND operator."""
        P, Q = Symbol("P Q")
        X, Y = Variable("X Y")

        expr = sp.And(P(X), Q(Y))
        variables = extract_variables(expr)

        # Should find both X and Y
        assert X in variables
        assert Y in variables

    def test_traverses_or_operator(self):
        """Test traversal through OR operator."""
        P, Q = Symbol("P Q")
        X, Y = Variable("X Y")

        expr = sp.Or(P(X), Q(Y))
        variables = extract_variables(expr)

        # Should find both X and Y
        assert X in variables
        assert Y in variables

    def test_traverses_not_operator(self):
        """Test traversal through NOT operator."""
        P = Symbol("P")
        X = Variable("X")

        expr = sp.Not(P(X))
        variables = extract_variables(expr)

        # Should find X inside NOT
        assert X in variables

    def test_traverses_multiple_levels(self):
        """Test traversal through multiple nesting levels."""
        P, Q, R = Symbol("P Q R")
        X, Y, Z = Variable("X Y Z")

        # sp.Not(sp.And(P(X), sp.Or(Q(Y), R(Z))))
        expr = sp.Not(sp.And(P(X), sp.Or(Q(Y), R(Z))))
        variables = extract_variables(expr)

        # Should find all three variables at different nesting levels
        assert len(variables) == 3
        assert X in variables
        assert Y in variables
        assert Z in variables
