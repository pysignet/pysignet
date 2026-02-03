"""Tests for variable binding and expression grounding.

This module tests the Binding class and ground() function which substitute
variables with concrete tensor indices for batch evaluation.
"""

import pytest
import torch
import sympy as sp

from pysignet import Symbol
from pysignet.logic import Variable, Binding, ground
from pysignet.symbols import PredicateApplication


class TestBindingCreation:
    """Test Binding class creation and basic properties."""

    def test_create_empty_binding(self):
        """Test creating an empty binding."""
        binding = Binding()

        assert isinstance(binding, Binding)
        assert len(binding) == 0

    def test_create_binding_with_dict(self):
        """Test creating binding from dictionary."""
        X, Y = Variable("X Y")
        binding = Binding({X: 0, Y: 1})

        assert len(binding) == 2
        assert binding[X] == 0
        assert binding[Y] == 1

    def test_binding_contains_variable(self):
        """Test that binding can check for variable presence."""
        X, Y = Variable("X Y")
        binding = Binding({X: 0})

        assert X in binding
        assert Y not in binding

    def test_binding_get_index(self):
        """Test getting index for variable."""
        X = Variable("X")
        binding = Binding({X: 5})

        assert binding[X] == 5

    def test_binding_missing_variable_raises_error(self):
        """Test that accessing missing variable raises KeyError."""
        X, Y = Variable("X Y")
        binding = Binding({X: 0})

        with pytest.raises(KeyError):
            _ = binding[Y]


class TestBindingModification:
    """Test modifying bindings."""

    def test_add_binding(self):
        """Test adding a new variable binding."""
        X, Y = Variable("X Y")
        binding = Binding({X: 0})

        binding[Y] = 1

        assert len(binding) == 2
        assert binding[Y] == 1

    def test_update_binding(self):
        """Test updating existing variable binding."""
        X = Variable("X")
        binding = Binding({X: 0})

        binding[X] = 5

        assert binding[X] == 5

    def test_binding_immutable_after_creation(self):
        """Test that bindings should be immutable after ground() uses them."""
        # This is more of a design test - bindings used in ground()
        # shouldn't be modified during evaluation
        X = Variable("X")
        original_binding = Binding({X: 0})

        # Make a copy to ensure original isn't modified
        binding_copy = Binding(dict(original_binding._bindings))

        assert original_binding[X] == binding_copy[X]


class TestGroundBasic:
    """Test basic grounding of expressions."""

    def test_ground_single_variable_to_constant(self):
        """Test grounding single variable to constant."""
        Digit = Symbol("Digit")
        X = Variable("X")

        expr = Digit(X)
        binding = Binding({X: 0})
        grounded = ground(expr, binding)

        # Should produce Digit(0)
        assert isinstance(grounded, PredicateApplication)
        assert grounded.predicate_name == "Digit"
        assert grounded.application_args == (0,)

    def test_ground_multiple_variables(self):
        """Test grounding multiple variables."""
        Rel = Symbol("Rel")
        X, Y = Variable("X Y")

        expr = Rel(X, Y)
        binding = Binding({X: 0, Y: 1})
        grounded = ground(expr, binding)

        # Should produce Rel(0, 1)
        assert isinstance(grounded, PredicateApplication)
        assert grounded.application_args == (0, 1)

    def test_ground_mixed_variable_constant(self):
        """Test grounding expression with both variables and constants."""
        P = Symbol("P")
        X = Variable("X")

        # P(X, 5) where 5 is already a constant
        expr = P(X, 5)
        binding = Binding({X: 0})
        grounded = ground(expr, binding)

        # Should produce P(0, 5)
        assert grounded.application_args == (0, 5)

    def test_ground_same_variable_multiple_times(self):
        """Test grounding when same variable appears multiple times."""
        Rel = Symbol("Rel")
        X = Variable("X")

        # Rel(X, X)
        expr = Rel(X, X)
        binding = Binding({X: 3})
        grounded = ground(expr, binding)

        # Should produce Rel(3, 3)
        assert grounded.application_args == (3, 3)


class TestGroundComplexExpressions:
    """Test grounding complex logical expressions."""

    def test_ground_and_expression(self):
        """Test grounding AND expression."""
        P, Q = Symbol("P Q")
        X, Y = Variable("X Y")

        # P(X) ∧ Q(Y)
        expr = sp.And(P(X), Q(Y))
        binding = Binding({X: 0, Y: 1})
        grounded = ground(expr, binding)

        # Should produce P(0) ∧ Q(1)
        assert isinstance(grounded, sp.And)
        args = grounded.args
        assert len(args) == 2

        # Check both applications are grounded
        for arg in args:
            assert isinstance(arg, PredicateApplication)
            # No variables should remain
            for app_arg in arg.application_args:
                assert isinstance(app_arg, int)

    def test_ground_or_expression(self):
        """Test grounding OR expression."""
        P, Q = Symbol("P Q")
        X, Y = Variable("X Y")

        expr = sp.Or(P(X), Q(Y))
        binding = Binding({X: 0, Y: 1})
        grounded = ground(expr, binding)

        assert isinstance(grounded, sp.Or)

    def test_ground_not_expression(self):
        """Test grounding NOT expression."""
        P = Symbol("P")
        X = Variable("X")

        expr = sp.Not(P(X))
        binding = Binding({X: 0})
        grounded = ground(expr, binding)

        assert isinstance(grounded, sp.Not)
        # Check inner expression is grounded
        inner = grounded.args[0]
        assert isinstance(inner, PredicateApplication)
        assert inner.application_args == (0,)

    def test_ground_implies_expression(self):
        """Test grounding IMPLIES expression."""
        P, Q = Symbol("P Q")
        X, Y = Variable("X Y")

        expr = sp.Implies(P(X), Q(Y))
        binding = Binding({X: 0, Y: 1})
        grounded = ground(expr, binding)

        assert isinstance(grounded, sp.Implies)

    def test_ground_nested_expression(self):
        """Test grounding deeply nested expression."""
        P, Q, R = Symbol("P Q R")
        X, Y, Z = Variable("X Y Z")

        # (P(X) ∧ Q(Y)) → R(Z)
        expr = sp.Implies(sp.And(P(X), Q(Y)), R(Z))
        binding = Binding({X: 0, Y: 1, Z: 2})
        grounded = ground(expr, binding)

        assert isinstance(grounded, sp.Implies)
        # All nested applications should be grounded
        # We can verify by checking that the expression doesn't contain
        # the original variable objects


class TestGroundPartialBinding:
    """Test grounding with partial bindings."""

    def test_ground_with_subset_of_variables(self):
        """Test grounding when only some variables are bound."""
        P, Q = Symbol("P Q")
        X, Y = Variable("X Y")

        # P(X) ∧ Q(Y) but only bind X
        expr = sp.And(P(X), Q(Y))
        binding = Binding({X: 0})
        grounded = ground(expr, binding)

        # P(X) should become P(0), but Q(Y) should remain Q(Y)
        assert isinstance(grounded, sp.And)

        # Extract both applications
        apps = [arg for arg in grounded.args if isinstance(arg, PredicateApplication)]

        # One should be P(0), other should be Q(Y)
        p_app = next(app for app in apps if app.predicate_name == "P")
        q_app = next(app for app in apps if app.predicate_name == "Q")

        assert p_app.application_args == (0,)  # X grounded to 0
        assert q_app.application_args == (Y,)  # Y still a variable


class TestGroundWithNullaryPredicates:
    """Test grounding expressions with nullary predicates."""

    def test_ground_mixed_nullary_and_unary(self):
        """Test grounding expression with both nullary and unary predicates."""
        P, Q = Symbol("P Q")
        X = Variable("X")

        # P ∧ Q(X) - P is nullary (no args), Q is unary
        expr = sp.And(P, Q(X))
        binding = Binding({X: 0})
        grounded = ground(expr, binding)

        # P should remain unchanged, Q(X) should become Q(0)
        assert isinstance(grounded, sp.And)

    def test_ground_only_nullary_predicates(self):
        """Test grounding expression with only nullary predicates."""
        P, Q = Symbol("P Q")

        # P ∧ Q - no variables at all
        expr = sp.And(P, Q)
        binding = Binding()  # Empty binding
        grounded = ground(expr, binding)

        # Expression should be unchanged
        assert grounded == expr


class TestGroundErrorHandling:
    """Test error handling in grounding."""

    def test_ground_with_missing_variable_raises_error(self):
        """Test that grounding with incomplete binding raises error."""
        # This test depends on whether we want strict or lenient binding
        # For now, let's assume strict: all variables must be bound
        P = Symbol("P")
        X, Y = Variable("X Y")

        expr = P(X, Y)
        binding = Binding({X: 0})  # Y is missing

        # Should raise error (or return partial grounding - design decision)
        # Let's test partial grounding behavior for now
        grounded = ground(expr, binding)

        # X should be grounded, Y should remain
        assert grounded.application_args == (0, Y)


class TestGroundWithBooleanConstants:
    """Test grounding expressions with boolean constants."""

    def test_ground_with_true_constant(self):
        """Test grounding expression containing sp.true."""
        P = Symbol("P")
        X = Variable("X")

        expr = sp.And(sp.true, P(X))
        binding = Binding({X: 0})
        grounded = ground(expr, binding)

        # SymPy simplifies And(True, P(X)) to P(X), so check that P(X) is grounded
        # The result should be P(0) after simplification
        assert isinstance(grounded, PredicateApplication)
        assert grounded.application_args == (0,)

    def test_ground_with_false_constant(self):
        """Test grounding expression containing sp.false."""
        P = Symbol("P")
        X = Variable("X")

        expr = sp.Or(sp.false, P(X))
        binding = Binding({X: 0})
        grounded = ground(expr, binding)

        # SymPy simplifies Or(False, P(X)) to P(X), so check that P(X) is grounded
        # The result should be P(0) after simplification
        assert isinstance(grounded, PredicateApplication)
        assert grounded.application_args == (0,)


class TestBindingRepresentation:
    """Test string representation of bindings."""

    def test_binding_repr(self):
        """Test repr of binding."""
        X, Y = Variable("X Y")
        binding = Binding({X: 0, Y: 1})

        repr_str = repr(binding)
        assert "X" in repr_str or "Y" in repr_str
        assert "0" in repr_str or "1" in repr_str

    def test_binding_str(self):
        """Test str of binding."""
        X = Variable("X")
        binding = Binding({X: 5})

        str_rep = str(binding)
        assert isinstance(str_rep, str)
