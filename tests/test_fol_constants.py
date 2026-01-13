"""Tests for constant extraction from logical expressions.

This module tests functions that identify and extract constants from
PredicateApplications, distinguishing them from variables.
"""

import pytest
import sympy as sp

from pysignet import Symbol
from pysignet.logic import Variable, extract_constants, is_constant, is_variable
from pysignet.logic.variable import VariableSymbol


class TestConstantDetection:
    """Test detection of constants vs variables in predicate arguments."""

    def test_integer_is_constant(self):
        """Test that integers are detected as constants."""
        assert is_constant(5) is True
        assert is_constant(0) is True
        assert is_constant(-10) is True

    def test_string_is_constant(self):
        """Test that strings are detected as constants."""
        assert is_constant("red") is True
        assert is_constant("Green") is True
        assert is_constant("") is True

    def test_float_is_constant(self):
        """Test that floats are detected as constants."""
        assert is_constant(3.14) is True
        assert is_constant(0.0) is True
        assert is_constant(-2.5) is True

    def test_none_is_constant(self):
        """Test that None is detected as a constant."""
        assert is_constant(None) is True

    def test_variable_is_not_constant(self):
        """Test that VariableSymbol is not a constant."""
        X = Variable("X")
        assert is_constant(X) is False

    def test_variable_detection(self):
        """Test detection of variables."""
        X = Variable("X")
        assert is_variable(X) is True

    def test_integer_is_not_variable(self):
        """Test that integers are not variables."""
        assert is_variable(5) is False
        assert is_variable(0) is False

    def test_string_is_not_variable(self):
        """Test that strings are not variables."""
        assert is_variable("red") is False

    def test_tuple_constant(self):
        """Test that tuples are detected as constants."""
        assert is_constant((1, 2, 3)) is True
        assert is_constant(("a", "b")) is True


class TestBasicConstantExtraction:
    """Test basic constant extraction from simple expressions."""

    def test_extract_from_single_constant_arg(self):
        """Test extracting single constant from predicate application."""
        Digit = Symbol("Digit")

        expr = Digit(5)
        constants = extract_constants(expr)

        assert len(constants) == 1
        assert 5 in constants

    def test_extract_from_multiple_constant_args(self):
        """Test extracting multiple constants from one predicate."""
        Rel = Symbol("Rel")

        # Rel(0, 1, 2)
        expr = Rel(0, 1, 2)
        constants = extract_constants(expr)

        assert len(constants) == 3
        assert 0 in constants
        assert 1 in constants
        assert 2 in constants

    def test_extract_string_constants(self):
        """Test extracting string constants."""
        Color = Symbol("Color")

        expr = Color("red")
        constants = extract_constants(expr)

        assert len(constants) == 1
        assert "red" in constants

    def test_extract_mixed_constant_types(self):
        """Test extracting different types of constants."""
        P = Symbol("P")

        # P(5, "label", 3.14)
        expr = P(5, "label", 3.14)
        constants = extract_constants(expr)

        assert len(constants) == 3
        assert 5 in constants
        assert "label" in constants
        assert 3.14 in constants

    def test_no_constants_with_variables_only(self):
        """Test that variables are not extracted as constants."""
        Digit = Symbol("Digit")
        X = Variable("X")

        expr = Digit(X)
        constants = extract_constants(expr)

        assert len(constants) == 0

    def test_extract_from_mixed_variable_and_constant(self):
        """Test extracting constants when mixed with variables."""
        P = Symbol("P")
        X, Y = Variable("X Y")

        # P(X, 5, Y, "red") - should only extract 5 and "red"
        expr = P(X, 5, Y, "red")
        constants = extract_constants(expr)

        assert len(constants) == 2
        assert 5 in constants
        assert "red" in constants
        # Variables should not be in constants
        assert X not in constants
        assert Y not in constants


class TestComplexExpressions:
    """Test constant extraction from complex logical expressions."""

    def test_extract_from_and_expression(self):
        """Test extraction from AND of multiple predicates."""
        P, Q = Symbol("P Q")

        # P(0) ∧ Q(1)
        expr = sp.And(P(0), Q(1))
        constants = extract_constants(expr)

        assert len(constants) == 2
        assert 0 in constants
        assert 1 in constants

    def test_extract_from_nested_and_or(self):
        """Test extraction from nested AND/OR expressions."""
        P, Q, R = Symbol("P Q R")

        # (P(0) ∧ Q(1)) ∨ R(2)
        expr = sp.Or(sp.And(P(0), Q(1)), R(2))
        constants = extract_constants(expr)

        assert len(constants) == 3
        assert 0 in constants
        assert 1 in constants
        assert 2 in constants

    def test_extract_duplicate_constants(self):
        """Test that duplicate constants are only extracted once."""
        P, Q, R = Symbol("P Q R")

        # P(5) ∧ Q(5) ∧ R(5) - constant 5 appears three times
        expr = sp.And(P(5), Q(5), R(5))
        constants = extract_constants(expr)

        assert len(constants) == 1
        assert 5 in constants

    def test_extract_from_implication(self):
        """Test extraction from implication."""
        P, Q = Symbol("P Q")

        # P(0) → Q(1)
        expr = sp.Implies(P(0), Q(1))
        constants = extract_constants(expr)

        assert len(constants) == 2
        assert 0 in constants
        assert 1 in constants

    def test_extract_from_negation(self):
        """Test extraction from negation."""
        P = Symbol("P")

        # ¬P(42)
        expr = sp.Not(P(42))
        constants = extract_constants(expr)

        assert len(constants) == 1
        assert 42 in constants

    def test_extract_from_deeply_nested_expression(self):
        """Test extraction from deeply nested expression."""
        P, Q, R, S = Symbol("P Q R S")
        X = Variable("X")

        # ((P(X, 0) ∧ Q(1)) → R(2)) ∨ S(3)
        expr = sp.Or(
            sp.Implies(sp.And(P(X, 0), Q(1)), R(2)),
            S(3)
        )
        constants = extract_constants(expr)

        assert len(constants) == 4
        assert 0 in constants
        assert 1 in constants
        assert 2 in constants
        assert 3 in constants
        # Variable X should not be extracted
        assert X not in constants


class TestEdgeCases:
    """Test edge cases for constant extraction."""

    def test_extract_from_nullary_predicate(self):
        """Test extraction from nullary predicate (no args)."""
        P = Symbol("P")

        # P - no constants
        expr = P
        constants = extract_constants(expr)

        assert len(constants) == 0

    def test_extract_from_boolean_constant(self):
        """Test extraction from boolean constant."""
        # sp.true - no constants in predicate args
        expr = sp.true
        constants = extract_constants(expr)

        assert len(constants) == 0

    def test_extract_zero_constant(self):
        """Test that zero is properly extracted as a constant."""
        P = Symbol("P")

        expr = P(0)
        constants = extract_constants(expr)

        assert len(constants) == 1
        assert 0 in constants

    def test_extract_empty_string(self):
        """Test that empty string is properly extracted."""
        P = Symbol("P")

        expr = P("")
        constants = extract_constants(expr)

        assert len(constants) == 1
        assert "" in constants

    def test_extract_negative_numbers(self):
        """Test extraction of negative number constants."""
        P = Symbol("P")

        expr = P(-5, -10)
        constants = extract_constants(expr)

        assert len(constants) == 2
        assert -5 in constants
        assert -10 in constants


class TestReturnType:
    """Test the return type and properties of extracted constants."""

    def test_returns_set(self):
        """Test that extract_constants returns a set."""
        P = Symbol("P")

        expr = P(5)
        constants = extract_constants(expr)

        assert isinstance(constants, set)

    def test_returns_empty_set_for_no_constants(self):
        """Test that empty set is returned when no constants."""
        P = Symbol("P")
        X = Variable("X")

        # Only variables, no constants
        expr = P(X)
        constants = extract_constants(expr)

        assert isinstance(constants, set)
        assert len(constants) == 0

    def test_set_uniqueness(self):
        """Test that set ensures uniqueness of constants."""
        P, Q, R = Symbol("P Q R")

        # Same constant multiple times
        expr = sp.And(P(7), sp.And(Q(7), R(7)))
        constants = extract_constants(expr)

        # Should only contain one instance of 7
        assert len(constants) == 1
        assert 7 in constants


class TestRealWorldPatterns:
    """Test real-world usage patterns."""

    def test_transitivity_constraint_with_constants(self):
        """Test extraction from transitivity constraint with string constants."""
        E = Symbol("E")

        # E("P", "H") ∧ E("H", "G") → E("P", "G")
        # This is the pattern used in ConsistencyChecker tests
        expr = sp.Implies(
            sp.And(E("P", "H"), E("H", "G")),
            E("P", "G")
        )
        constants = extract_constants(expr)

        assert len(constants) == 3
        assert "P" in constants
        assert "H" in constants
        assert "G" in constants

    def test_color_constraint_with_mixed_args(self):
        """Test extraction from color classification with mixed args."""
        Digit = Symbol("Digit")
        Color = Symbol("Color")
        X = Variable("X")

        # Digit(X, 5) ∧ Color(X, "red")
        expr = sp.And(Digit(X, 5), Color(X, "red"))
        constants = extract_constants(expr)

        assert len(constants) == 2
        assert 5 in constants
        assert "red" in constants
        assert X not in constants

    def test_relation_with_multiple_constants(self):
        """Test extraction from relation with multiple constant arguments."""
        Between = Symbol("Between")

        # Between(1, 5, 10) - all constants
        expr = Between(1, 5, 10)
        constants = extract_constants(expr)

        assert len(constants) == 3
        assert 1 in constants
        assert 5 in constants
        assert 10 in constants

    def test_no_constants_in_variable_only_expression(self):
        """Test that expressions with only variables extract no constants."""
        P, Q = Symbol("P Q")
        X, Y, Z = Variable("X Y Z")

        # Complex expression but only variables
        expr = sp.And(P(X, Y), Q(Y, Z))
        constants = extract_constants(expr)

        assert len(constants) == 0
