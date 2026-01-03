"""Tests for FOL quantifier classes (ForAll, Exists).

This module tests the quantifier classes that support domain specification
for first-order logic expressions.

Each quantifier binds a single variable to values from a domain.
Multiple variables are handled via nesting.
"""

import pytest
import sympy as sp

from pysignet.logic import Variable
from pysignet.logic.quantifier import ForAll, Exists
from pysignet import Symbol


class TestForAllBasics:
    """Tests for ForAll quantifier basic functionality."""

    def test_forall_creation_with_list_domain(self):
        """ForAll can be created with a list domain."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        domain = [0, 1, 2]
        forall = ForAll(X, domain, expr)

        assert forall is not None
        assert forall.variable == X
        assert forall.domain == domain
        assert forall.body == expr

    def test_forall_creation_with_range_domain(self):
        """ForAll can be created with a range domain."""
        Y = Variable("Y")
        Q = Symbol("Q")
        expr = Q(Y)

        domain = range(5)
        forall = ForAll(Y, domain, expr)

        assert forall is not None
        assert forall.variable == Y
        assert forall.domain == domain
        assert forall.body == expr

    def test_forall_creation_with_tuple_domain(self):
        """ForAll can be created with a tuple domain."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        domain = (0, 2, 4, 6, 8)
        forall = ForAll(X, domain, expr)

        assert forall.domain == domain

    def test_forall_creation_with_set_domain(self):
        """ForAll can be created with a set domain."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        # Note: sets are unordered, but should work
        domain = {0, 1, 2}
        forall = ForAll(X, domain, expr)

        assert forall.domain == domain

    def test_forall_binds_single_variable(self):
        """ForAll binds a single variable."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        forall = ForAll(X, [0, 1], expr)

        assert forall.variable == X

    def test_forall_string_representation(self):
        """ForAll has proper string representation."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        forall = ForAll(X, [0, 1, 2], expr)

        str_repr = str(forall)
        assert "ForAll" in str_repr or "∀" in str_repr

    def test_forall_with_complex_domain(self):
        """ForAll works with domains of different types."""
        X = Variable("X")
        P = Symbol("P")

        # String domain
        forall1 = ForAll(X, ["red", "green", "blue"], P(X))
        assert forall1.domain == ["red", "green", "blue"]

        # Mixed type domain (though unusual)
        forall2 = ForAll(X, [0, "one", 2.0], P(X))
        assert forall2.domain == [0, "one", 2.0]


class TestExistsBasics:
    """Tests for Exists quantifier basic functionality."""

    def test_exists_creation_with_list_domain(self):
        """Exists can be created with a list domain."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        domain = [0, 1, 2]
        exists = Exists(X, domain, expr)

        assert exists is not None
        assert exists.variable == X
        assert exists.domain == domain
        assert exists.body == expr

    def test_exists_creation_with_range_domain(self):
        """Exists can be created with a range domain."""
        Y = Variable("Y")
        Q = Symbol("Q")
        expr = Q(Y)

        domain = range(10)
        exists = Exists(Y, domain, expr)

        assert exists is not None
        assert exists.variable == Y
        assert exists.domain == domain
        assert exists.body == expr

    def test_exists_binds_single_variable(self):
        """Exists binds a single variable."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        exists = Exists(X, [0, 1, 2], expr)

        assert exists.variable == X

    def test_exists_string_representation(self):
        """Exists has proper string representation."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        exists = Exists(X, [0, 1], expr)

        str_repr = str(exists)
        assert "Exists" in str_repr or "∃" in str_repr


class TestVariableScoping:
    """Tests for variable scoping in quantifiers."""

    def test_forall_variable_is_bound(self):
        """Variable in ForAll is considered bound."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        forall = ForAll(X, [0, 1, 2], expr)

        # Variable should be accessible
        assert forall.variable == X

    def test_free_variables_not_in_quantifier(self):
        """Free variables in body are not the quantified variable."""
        X, Y = Variable("X Y")
        P = Symbol("P")
        expr = P(X, Y)

        # Only quantify X, Y is free
        forall = ForAll(X, [0, 1], expr)

        assert forall.variable == X
        # Y appears in the body but is not bound by this quantifier

    def test_nested_quantifier_scoping(self):
        """Nested quantifiers maintain proper variable scoping."""
        X, Y = Variable("X Y")
        P = Symbol("P")
        inner_expr = P(X, Y)

        # Nested: ForAll X. ForAll Y. P(X, Y)
        inner = ForAll(Y, [0, 1], inner_expr)
        outer = ForAll(X, [2, 3], inner)

        assert outer.variable == X
        assert outer.body == inner
        assert inner.variable == Y


class TestNestedQuantifiers:
    """Tests for nested quantifier structures."""

    def test_forall_forall_nesting(self):
        """ForAll can be nested inside ForAll."""
        X, Y = Variable("X Y")
        P = Symbol("P")

        inner = ForAll(Y, [0, 1], P(X, Y))
        outer = ForAll(X, [2, 3], inner)

        assert isinstance(outer, ForAll)
        assert isinstance(outer.body, ForAll)
        assert outer.variable == X
        assert inner.variable == Y

    def test_exists_exists_nesting(self):
        """Exists can be nested inside Exists."""
        X, Y = Variable("X Y")
        P = Symbol("P")

        inner = Exists(Y, [0, 1], P(X, Y))
        outer = Exists(X, [2, 3], inner)

        assert isinstance(outer, Exists)
        assert isinstance(outer.body, Exists)
        assert outer.variable == X
        assert inner.variable == Y

    def test_forall_exists_nesting(self):
        """ForAll can contain Exists."""
        X, Y = Variable("X Y")
        P = Symbol("P")

        inner = Exists(Y, [0, 1, 2], P(X, Y))
        outer = ForAll(X, [3, 4], inner)

        assert isinstance(outer, ForAll)
        assert isinstance(outer.body, Exists)

    def test_exists_forall_nesting(self):
        """Exists can contain ForAll."""
        X, Y = Variable("X Y")
        P = Symbol("P")

        inner = ForAll(Y, [0, 1], P(X, Y))
        outer = Exists(X, [2, 3, 4], inner)

        assert isinstance(outer, Exists)
        assert isinstance(outer.body, ForAll)

    def test_triple_nesting(self):
        """Quantifiers can be nested three levels deep."""
        X, Y, Z = Variable("X Y Z")
        P = Symbol("P")

        innermost = P(X, Y, Z)
        level2 = Exists(Z, [0, 1], innermost)
        level1 = ForAll(Y, [2, 3], level2)
        outermost = ForAll(X, [4, 5], level1)

        assert isinstance(outermost, ForAll)
        assert isinstance(outermost.body, ForAll)
        assert isinstance(outermost.body.body, Exists)
        assert outermost.variable == X
        assert level1.variable == Y
        assert level2.variable == Z


class TestComplexExpressions:
    """Tests for quantifiers with complex logical expressions."""

    def test_forall_with_and(self):
        """ForAll with conjunction in body."""
        X, Y = Variable("X Y")
        P, Q = Symbol("P Q")

        expr = sp.And(P(X), Q(Y))
        forall = ForAll(X, [0, 1, 2], expr)

        assert isinstance(forall.body, sp.And)

    def test_forall_with_implies(self):
        """ForAll with implication in body."""
        X, Y = Variable("X Y")
        P, Q = Symbol("P Q")

        expr = sp.Implies(P(X, Y), Q(X))
        forall = ForAll(Y, [0, 1, 2], expr)

        assert isinstance(forall.body, sp.Implies)

    def test_exists_with_or(self):
        """Exists with disjunction in body."""
        X = Variable("X")
        P, Q = Symbol("P Q")

        expr = sp.Or(P(X), Q(X))
        exists = Exists(X, range(5), expr)

        assert isinstance(exists.body, sp.Or)

    def test_exists_with_not(self):
        """Exists with negation in body."""
        X = Variable("X")
        P = Symbol("P")

        expr = sp.Not(P(X))
        exists = Exists(X, [1, 2, 3], expr)

        assert isinstance(exists.body, sp.Not)

    def test_forall_with_free_and_bound_variables(self):
        """ForAll where body has both bound and free variables."""
        X, Y = Variable("X Y")
        P = Symbol("P")

        # Quantify only Y, X remains free
        expr = P(X, Y)
        forall = ForAll(Y, [0, 1, 2], expr)

        assert forall.variable == Y
        # X is free in the body


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_forall_with_empty_domain(self):
        """ForAll with empty domain."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        forall = ForAll(X, [], expr)

        # Should create successfully
        assert forall.domain == []

    def test_exists_with_empty_domain(self):
        """Exists with empty domain."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)

        exists = Exists(X, [], expr)

        # Should create successfully
        assert exists.domain == []

    def test_forall_with_single_element_domain(self):
        """ForAll with single element domain."""
        X = Variable("X")
        P = Symbol("P")

        forall = ForAll(X, [42], P(X))

        assert len(list(forall.domain)) == 1

    def test_exists_with_single_element_domain(self):
        """Exists with single element domain."""
        X = Variable("X")
        P = Symbol("P")

        exists = Exists(X, [42], P(X))

        assert len(list(exists.domain)) == 1


class TestSymPyCompatibility:
    """Tests for SymPy compatibility."""

    def test_forall_is_sympy_basic(self):
        """ForAll inherits from sp.Basic."""
        X = Variable("X")
        P = Symbol("P")

        forall = ForAll(X, [0, 1], P(X))

        assert isinstance(forall, sp.Basic)

    def test_exists_is_sympy_basic(self):
        """Exists inherits from sp.Basic."""
        X = Variable("X")
        P = Symbol("P")

        exists = Exists(X, [0, 1], P(X))

        assert isinstance(exists, sp.Basic)

    def test_forall_args_property(self):
        """ForAll has proper args property for SymPy."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)
        domain = [0, 1, 2]

        forall = ForAll(X, domain, expr)

        # Should have args property (SymPy requirement)
        assert hasattr(forall, 'args')
        # Args should be a tuple
        assert isinstance(forall.args, tuple)

    def test_exists_args_property(self):
        """Exists has proper args property for SymPy."""
        X = Variable("X")
        P = Symbol("P")
        expr = P(X)
        domain = [0, 1, 2]

        exists = Exists(X, domain, expr)

        # Should have args property (SymPy requirement)
        assert hasattr(exists, 'args')
        assert isinstance(exists.args, tuple)


class TestEquality:
    """Tests for quantifier equality."""

    def test_forall_equality_same_structure(self):
        """ForAll instances with same structure are equal."""
        X = Variable("X")
        P = Symbol("P")

        forall1 = ForAll(X, [0, 1, 2], P(X))
        forall2 = ForAll(X, [0, 1, 2], P(X))

        # SymPy equality
        assert forall1 == forall2

    def test_exists_equality_same_structure(self):
        """Exists instances with same structure are equal."""
        X = Variable("X")
        P = Symbol("P")

        exists1 = Exists(X, [0, 1], P(X))
        exists2 = Exists(X, [0, 1], P(X))

        assert exists1 == exists2

    def test_forall_inequality_different_domain(self):
        """ForAll instances with different domains are not equal."""
        X = Variable("X")
        P = Symbol("P")

        forall1 = ForAll(X, [0, 1, 2], P(X))
        forall2 = ForAll(X, [0, 1], P(X))

        assert forall1 != forall2

    def test_forall_inequality_different_variable(self):
        """ForAll instances with different variables are not equal."""
        X, Y = Variable("X Y")
        P = Symbol("P")

        forall1 = ForAll(X, [0, 1], P(X))
        forall2 = ForAll(Y, [0, 1], P(Y))

        assert forall1 != forall2

    def test_forall_exists_not_equal(self):
        """ForAll and Exists are not equal even with same structure."""
        X = Variable("X")
        P = Symbol("P")

        forall = ForAll(X, [0, 1], P(X))
        exists = Exists(X, [0, 1], P(X))

        assert forall != exists


class TestRealWorldPatterns:
    """Tests for common real-world usage patterns."""

    def test_one_hot_constraint_pattern(self):
        """Common pattern: exactly one class (one-hot)."""
        X, Y = Variable("X Y")
        Digit = Symbol("Digit")

        # ∃Y ∈ {0..9}. Digit(X, Y)
        # "X is classified as exactly one digit"
        expr = Exists(Y, range(10), Digit(X, Y))

        assert isinstance(expr, Exists)
        assert expr.variable == Y
        assert expr.domain == range(10)

    def test_even_digits_constraint_pattern(self):
        """Common pattern: constraint on subset of classes."""
        X, Y = Variable("X Y")
        Digit, Even = Symbol("Digit Even")

        # ∀Y ∈ {0,2,4,6,8}. Digit(X, Y) → Even(X)
        # "For all even digit labels, if X is that digit, then X is even"
        expr = ForAll(Y, [0, 2, 4, 6, 8], sp.Implies(Digit(X, Y), Even(X)))

        assert isinstance(expr, ForAll)
        assert expr.variable == Y
        assert expr.domain == [0, 2, 4, 6, 8]

    def test_multi_label_classification_pattern(self):
        """Pattern: at least one label from set."""
        X, Y = Variable("X Y")
        Label = Symbol("Label")

        # ∃Y ∈ {cat, dog, bird}. Label(X, Y)
        # "X has at least one of these labels"
        expr = Exists(Y, ["cat", "dog", "bird"], Label(X, Y))

        assert isinstance(expr, Exists)
        assert expr.domain == ["cat", "dog", "bird"]
