"""Tests for FOL quantifier expansion over domains.

This module tests the expansion of ForAll and Exists quantifiers over
their specified domains into conjunctions and disjunctions.
"""

import pytest
import sympy as sp

from pysignet.logic import Variable, ForAll, Exists
from pysignet.logic.expansion import expand_quantifier
from pysignet import Symbol


class TestForAllExpansion:
    """Tests for ForAll quantifier expansion."""

    def test_forall_expands_to_conjunction(self):
        """ForAll expands to conjunction over domain."""
        Y = Variable("Y")
        P = Symbol("P")

        # ForAll(Y, [0, 1, 2], P(Y))
        forall = ForAll(Y, [0, 1, 2], P(Y))
        expanded = expand_quantifier(forall)

        # Should expand to: P(0) ∧ P(1) ∧ P(2)
        expected = sp.And(P(0), P(1), P(2))
        assert expanded == expected

    def test_forall_with_list_domain(self):
        """ForAll works with list domain."""
        Y = Variable("Y")
        P = Symbol("P")

        forall = ForAll(Y, [0, 2, 4], P(Y))
        expanded = expand_quantifier(forall)

        # P(0) ∧ P(2) ∧ P(4)
        expected = sp.And(P(0), P(2), P(4))
        assert expanded == expected

    def test_forall_with_range_domain(self):
        """ForAll works with range domain."""
        Y = Variable("Y")
        P = Symbol("P")

        forall = ForAll(Y, range(3), P(Y))
        expanded = expand_quantifier(forall)

        # P(0) ∧ P(1) ∧ P(2)
        expected = sp.And(P(0), P(1), P(2))
        assert expanded == expected

    def test_forall_with_tuple_domain(self):
        """ForAll works with tuple domain."""
        Y = Variable("Y")
        P = Symbol("P")

        forall = ForAll(Y, (5, 10, 15), P(Y))
        expanded = expand_quantifier(forall)

        expected = sp.And(P(5), P(10), P(15))
        assert expanded == expected

    def test_forall_with_string_domain(self):
        """ForAll works with string values in domain."""
        Y = Variable("Y")
        Color = Symbol("Color")

        forall = ForAll(Y, ["red", "green", "blue"], Color(Y))
        expanded = expand_quantifier(forall)

        expected = sp.And(Color("red"), Color("green"), Color("blue"))
        assert expanded == expected

    def test_forall_with_single_element_domain(self):
        """ForAll with single element domain."""
        Y = Variable("Y")
        P = Symbol("P")

        forall = ForAll(Y, [42], P(Y))
        expanded = expand_quantifier(forall)

        # Single element: just P(42)
        expected = P(42)
        assert expanded == expected

    def test_forall_with_empty_domain(self):
        """ForAll with empty domain returns true."""
        Y = Variable("Y")
        P = Symbol("P")

        forall = ForAll(Y, [], P(Y))
        expanded = expand_quantifier(forall)

        # Empty domain: vacuously true
        assert expanded == sp.true

    def test_forall_with_complex_body(self):
        """ForAll expands complex body expression."""
        Y = Variable("Y")
        P, Q = Symbol("P Q")

        # ForAll(Y, [0, 1], P(Y) → Q(Y))
        body = sp.Implies(P(Y), Q(Y))
        forall = ForAll(Y, [0, 1], body)
        expanded = expand_quantifier(forall)

        # (P(0) → Q(0)) ∧ (P(1) → Q(1))
        expected = sp.And(
            sp.Implies(P(0), Q(0)),
            sp.Implies(P(1), Q(1))
        )
        assert expanded == expected


class TestExistsExpansion:
    """Tests for Exists quantifier expansion."""

    def test_exists_expands_to_disjunction(self):
        """Exists expands to disjunction over domain."""
        Y = Variable("Y")
        P = Symbol("P")

        # Exists(Y, [0, 1, 2], P(Y))
        exists = Exists(Y, [0, 1, 2], P(Y))
        expanded = expand_quantifier(exists)

        # Should expand to: P(0) ∨ P(1) ∨ P(2)
        expected = sp.Or(P(0), P(1), P(2))
        assert expanded == expected

    def test_exists_with_list_domain(self):
        """Exists works with list domain."""
        Y = Variable("Y")
        P = Symbol("P")

        exists = Exists(Y, [0, 2, 4], P(Y))
        expanded = expand_quantifier(exists)

        # P(0) ∨ P(2) ∨ P(4)
        expected = sp.Or(P(0), P(2), P(4))
        assert expanded == expected

    def test_exists_with_range_domain(self):
        """Exists works with range domain."""
        Y = Variable("Y")
        P = Symbol("P")

        exists = Exists(Y, range(10), P(Y))
        expanded = expand_quantifier(exists)

        # P(0) ∨ P(1) ∨ ... ∨ P(9)
        expected = sp.Or(*[P(i) for i in range(10)])
        assert expanded == expected

    def test_exists_with_single_element_domain(self):
        """Exists with single element domain."""
        Y = Variable("Y")
        P = Symbol("P")

        exists = Exists(Y, [42], P(Y))
        expanded = expand_quantifier(exists)

        # Single element: just P(42)
        expected = P(42)
        assert expanded == expected

    def test_exists_with_empty_domain(self):
        """Exists with empty domain returns false."""
        Y = Variable("Y")
        P = Symbol("P")

        exists = Exists(Y, [], P(Y))
        expanded = expand_quantifier(exists)

        # Empty domain: false
        assert expanded == sp.false

    def test_exists_with_complex_body(self):
        """Exists expands complex body expression."""
        Y = Variable("Y")
        P, Q = Symbol("P Q")

        # Exists(Y, [0, 1], P(Y) ∧ Q(Y))
        body = sp.And(P(Y), Q(Y))
        exists = Exists(Y, [0, 1], body)
        expanded = expand_quantifier(exists)

        # (P(0) ∧ Q(0)) ∨ (P(1) ∧ Q(1))
        expected = sp.Or(
            sp.And(P(0), Q(0)),
            sp.And(P(1), Q(1))
        )
        assert expanded == expected


class TestNestedQuantifiers:
    """Tests for nested quantifier expansion."""

    def test_forall_forall_nesting(self):
        """Nested ForAll quantifiers expand correctly."""
        X, Y = Variable("X Y")
        P = Symbol("P")

        # ForAll(X, [0, 1], ForAll(Y, [2, 3], P(X, Y)))
        inner = ForAll(Y, [2, 3], P(X, Y))
        outer = ForAll(X, [0, 1], inner)

        expanded = expand_quantifier(outer)

        # Should expand to conjunction of expanded inner quantifiers
        # (P(0,2) ∧ P(0,3)) ∧ (P(1,2) ∧ P(1,3))
        inner_0 = sp.And(P(0, 2), P(0, 3))
        inner_1 = sp.And(P(1, 2), P(1, 3))
        expected = sp.And(inner_0, inner_1)

        assert expanded == expected

    def test_exists_exists_nesting(self):
        """Nested Exists quantifiers expand correctly."""
        X, Y = Variable("X Y")
        P = Symbol("P")

        # Exists(X, [0, 1], Exists(Y, [2, 3], P(X, Y)))
        inner = Exists(Y, [2, 3], P(X, Y))
        outer = Exists(X, [0, 1], inner)

        expanded = expand_quantifier(outer)

        # Should expand to disjunction of expanded inner quantifiers
        # (P(0,2) ∨ P(0,3)) ∨ (P(1,2) ∨ P(1,3))
        inner_0 = sp.Or(P(0, 2), P(0, 3))
        inner_1 = sp.Or(P(1, 2), P(1, 3))
        expected = sp.Or(inner_0, inner_1)

        assert expanded == expected

    def test_forall_exists_nesting(self):
        """ForAll containing Exists expands correctly."""
        X, Y = Variable("X Y")
        P = Symbol("P")

        # ForAll(X, [0, 1], Exists(Y, [2, 3], P(X, Y)))
        inner = Exists(Y, [2, 3], P(X, Y))
        outer = ForAll(X, [0, 1], inner)

        expanded = expand_quantifier(outer)

        # (P(0,2) ∨ P(0,3)) ∧ (P(1,2) ∨ P(1,3))
        inner_0 = sp.Or(P(0, 2), P(0, 3))
        inner_1 = sp.Or(P(1, 2), P(1, 3))
        expected = sp.And(inner_0, inner_1)

        assert expanded == expected

    def test_exists_forall_nesting(self):
        """Exists containing ForAll expands correctly."""
        X, Y = Variable("X Y")
        P = Symbol("P")

        # Exists(X, [0, 1], ForAll(Y, [2, 3], P(X, Y)))
        inner = ForAll(Y, [2, 3], P(X, Y))
        outer = Exists(X, [0, 1], inner)

        expanded = expand_quantifier(outer)

        # (P(0,2) ∧ P(0,3)) ∨ (P(1,2) ∧ P(1,3))
        inner_0 = sp.And(P(0, 2), P(0, 3))
        inner_1 = sp.And(P(1, 2), P(1, 3))
        expected = sp.Or(inner_0, inner_1)

        assert expanded == expected

    def test_triple_nested_quantifiers(self):
        """Triple nested quantifiers expand correctly."""
        X, Y, Z = Variable("X Y Z")
        P = Symbol("P")

        # ForAll(X, [0, 1], Exists(Y, [2], ForAll(Z, [3, 4], P(X, Y, Z))))
        innermost = ForAll(Z, [3, 4], P(X, Y, Z))
        middle = Exists(Y, [2], innermost)
        outer = ForAll(X, [0, 1], middle)

        expanded = expand_quantifier(outer)

        # ((P(0,2,3) ∧ P(0,2,4))) ∧ ((P(1,2,3) ∧ P(1,2,4)))
        z_expansion_0_2 = sp.And(P(0, 2, 3), P(0, 2, 4))
        z_expansion_1_2 = sp.And(P(1, 2, 3), P(1, 2, 4))
        expected = sp.And(z_expansion_0_2, z_expansion_1_2)

        assert expanded == expected


class TestVariableSubstitution:
    """Tests for correct variable substitution during expansion."""

    def test_variable_substituted_in_nested_expression(self):
        """Variable is correctly substituted in nested expressions."""
        Y = Variable("Y")
        P, Q = Symbol("P Q")

        # ForAll(Y, [0, 1], (P(Y) ∧ Q(Y)) → P(Y))
        body = sp.Implies(sp.And(P(Y), Q(Y)), P(Y))
        forall = ForAll(Y, [0, 1], body)
        expanded = expand_quantifier(forall)

        # ((P(0) ∧ Q(0)) → P(0)) ∧ ((P(1) ∧ Q(1)) → P(1))
        expected = sp.And(
            sp.Implies(sp.And(P(0), Q(0)), P(0)),
            sp.Implies(sp.And(P(1), Q(1)), P(1))
        )
        assert expanded == expected

    def test_free_variables_preserved(self):
        """Free variables (not quantified) are preserved."""
        X, Y = Variable("X Y")
        P = Symbol("P")

        # ForAll(Y, [0, 1], P(X, Y))
        # X is free, Y is bound
        forall = ForAll(Y, [0, 1], P(X, Y))
        expanded = expand_quantifier(forall)

        # P(X, 0) ∧ P(X, 1)
        # X should remain as variable
        expected = sp.And(P(X, 0), P(X, 1))
        assert expanded == expected

    def test_multiple_occurrences_of_variable(self):
        """Variable appears multiple times in body."""
        Y = Variable("Y")
        P, Q = Symbol("P Q")

        # ForAll(Y, [0, 1], P(Y) ∧ Q(Y) ∧ P(Y))
        body = sp.And(P(Y), Q(Y), P(Y))
        forall = ForAll(Y, [0, 1], body)
        expanded = expand_quantifier(forall)

        # (P(0) ∧ Q(0) ∧ P(0)) ∧ (P(1) ∧ Q(1) ∧ P(1))
        expected = sp.And(
            sp.And(P(0), Q(0), P(0)),
            sp.And(P(1), Q(1), P(1))
        )
        assert expanded == expected


class TestEdgeCases:
    """Tests for edge cases in quantifier expansion."""

    def test_expand_non_quantifier_raises_error(self):
        """Attempting to expand non-quantifier raises error."""
        P = Symbol("P")

        with pytest.raises((TypeError, AttributeError)):
            expand_quantifier(P)

    def test_forall_preserves_order(self):
        """ForAll preserves domain order in expansion."""
        Y = Variable("Y")
        P = Symbol("P")

        # Domain order matters for reproducibility
        forall = ForAll(Y, [9, 5, 1], P(Y))
        expanded = expand_quantifier(forall)

        # Order should be 9, 5, 1
        expected = sp.And(P(9), P(5), P(1))
        assert expanded == expected

    def test_exists_preserves_order(self):
        """Exists preserves domain order in expansion."""
        Y = Variable("Y")
        P = Symbol("P")

        exists = Exists(Y, [9, 5, 1], P(Y))
        expanded = expand_quantifier(exists)

        # Order should be 9, 5, 1
        expected = sp.Or(P(9), P(5), P(1))
        assert expanded == expected


class TestRealWorldPatterns:
    """Tests for common real-world usage patterns."""

    def test_one_hot_constraint(self):
        """One-hot constraint: exactly one class."""
        X, Y = Variable("X Y")
        Digit = Symbol("Digit")

        # Exists(Y, range(10), Digit(X, Y))
        # "X is classified as some digit"
        exists = Exists(Y, range(10), Digit(X, Y))
        expanded = expand_quantifier(exists)

        # Digit(X,0) ∨ Digit(X,1) ∨ ... ∨ Digit(X,9)
        expected = sp.Or(*[Digit(X, i) for i in range(10)])
        assert expanded == expected

    def test_even_digits_constraint(self):
        """Constraint on subset of classes."""
        X, Y = Variable("X Y")
        Digit, Even = Symbol("Digit Even")

        # ForAll(Y, [0,2,4,6,8], Digit(X, Y) → Even(X))
        body = sp.Implies(Digit(X, Y), Even(X))
        forall = ForAll(Y, [0, 2, 4, 6, 8], body)
        expanded = expand_quantifier(forall)

        # Conjunction of implications for each even digit
        expected = sp.And(*[
            sp.Implies(Digit(X, i), Even(X))
            for i in [0, 2, 4, 6, 8]
        ])
        assert expanded == expected
