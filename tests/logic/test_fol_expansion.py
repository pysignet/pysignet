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
        assert expanded.equals(expected)


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

        # ((P(0) ∧ Q(0)) -> P(0)) ∧ ((P(1) ∧ Q(1)) -> P(1))
        expected = sp.And(
            sp.Implies(sp.And(P(0), Q(0)), P(0)),
            sp.Implies(sp.And(P(1), Q(1)), P(1))
        )
        assert expanded.equals(expected)

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
        assert expanded.equals(expected)


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


class TestMultiVariableExpansion:
    """Tests for multi-variable quantifier expansion."""

    def test_forall_with_two_variables_expands(self):
        """ForAll with two variables expands over tuple domain."""
        I, J = Variable("I J")
        P = Symbol("P")

        pairs = [(0, 1), (0, 2), (1, 2)]
        forall = ForAll([I, J], pairs, P(I, J))
        expanded = expand_quantifier(forall)

        # P(0, 1) AND P(0, 2) AND P(1, 2)
        expected = sp.And(P(0, 1), P(0, 2), P(1, 2))
        assert expanded == expected

    def test_exists_with_two_variables_expands(self):
        """Exists with two variables expands over tuple domain."""
        I, J = Variable("I J")
        P = Symbol("P")

        pairs = [(0, 1), (1, 2)]
        exists = Exists([I, J], pairs, P(I, J))
        expanded = expand_quantifier(exists)

        # P(0, 1) OR P(1, 2)
        expected = sp.Or(P(0, 1), P(1, 2))
        assert expanded == expected

    def test_forall_multi_var_with_complex_body(self):
        """ForAll with multiple variables and complex body."""
        I, J, X = Variable("I J X")
        Digit = Symbol("Digit")

        pairs = [(0, 1), (0, 2)]
        # Mutual exclusivity: Digit(X, I) -> ~Digit(X, J)
        body = sp.Implies(Digit(X, I), sp.Not(Digit(X, J)))
        forall = ForAll([I, J], pairs, body)
        expanded = expand_quantifier(forall)

        # (Digit(X, 0) -> ~Digit(X, 1)) AND (Digit(X, 0) -> ~Digit(X, 2))
        expected = sp.And(
            sp.Implies(Digit(X, 0), sp.Not(Digit(X, 1))),
            sp.Implies(Digit(X, 0), sp.Not(Digit(X, 2)))
        )
        assert expanded == expected

    def test_forall_multi_var_single_element_domain(self):
        """ForAll with multiple variables and single element domain."""
        I, J = Variable("I J")
        P = Symbol("P")

        forall = ForAll([I, J], [(5, 10)], P(I, J))
        expanded = expand_quantifier(forall)

        # Single element: just P(5, 10)
        expected = P(5, 10)
        assert expanded == expected

    def test_forall_multi_var_empty_domain(self):
        """ForAll with multiple variables and empty domain."""
        I, J = Variable("I J")
        P = Symbol("P")

        forall = ForAll([I, J], [], P(I, J))
        expanded = expand_quantifier(forall)

        # Empty domain: vacuously true
        assert expanded == sp.true

    def test_exists_multi_var_empty_domain(self):
        """Exists with multiple variables and empty domain."""
        I, J = Variable("I J")
        P = Symbol("P")

        exists = Exists([I, J], [], P(I, J))
        expanded = expand_quantifier(exists)

        # Empty domain: false
        assert expanded == sp.false

    def test_at_most_one_constraint_expansion(self):
        """At-most-one constraint expands to pairwise implications."""
        X, I, J = Variable("X I J")
        Digit = Symbol("Digit")

        # All pairs (i, j) where i < j for 3 classes
        pairs = [(i, j) for i in range(3) for j in range(i + 1, 3)]
        # pairs = [(0, 1), (0, 2), (1, 2)]

        at_most_one = ForAll(
            [I, J], pairs, sp.Implies(Digit(X, I), sp.Not(Digit(X, J)))
        )
        expanded = expand_quantifier(at_most_one)

        # Should be conjunction of 3 implications
        expected = sp.And(
            sp.Implies(Digit(X, 0), sp.Not(Digit(X, 1))),
            sp.Implies(Digit(X, 0), sp.Not(Digit(X, 2))),
            sp.Implies(Digit(X, 1), sp.Not(Digit(X, 2)))
        )
        assert expanded == expected

    def test_multi_var_preserves_free_variables(self):
        """Multi-variable expansion preserves free variables."""
        X, I, J = Variable("X I J")
        P = Symbol("P")

        # X is free, I and J are bound
        pairs = [(0, 1), (1, 2)]
        forall = ForAll([I, J], pairs, P(X, I, J))
        expanded = expand_quantifier(forall)

        # X should remain, I and J should be substituted
        expected = sp.And(P(X, 0, 1), P(X, 1, 2))
        assert expanded == expected


class TestInternalHelpers:
    """Tests for internal _substitute_variable helper function."""

    def test_substitute_variable_in_nested_quantifier_shadowing(self):
        """Variable substitution stops at quantifier that shadows the variable."""
        from pysignet.logic.expansion import _substitute_variable

        X, Y = Variable("X Y")
        P = Symbol("P")

        # Nested quantifier where inner binds X (shadows outer X)
        inner = ForAll(X, [1, 2], P(X))  # This X is bound by ForAll

        # Try to substitute X -> 5 in the ForAll expression
        # Should NOT substitute inside because ForAll binds X
        result = _substitute_variable(inner, X, 5)

        # Result should be unchanged (ForAll blocks substitution)
        assert result == inner

    def test_substitute_variable_in_leaf(self):
        """Variable substitution works on leaf variable."""
        from pysignet.logic.expansion import _substitute_variable

        X = Variable("X")

        # Substitute X with 5 where X is a leaf
        result = _substitute_variable(X, X, 5)

        # Should return the value
        assert result == 5

    def test_substitute_variable_in_arithmetic_predicate_arg(self):
        """Substitution into arithmetic expression in predicate argument."""
        import sympy as sp
        from pysignet.logic.expansion import _substitute_in_predicate_application

        S, I = Variable("S I")
        Digit = Symbol("Digit")
        X = Variable("X")

        # Digit(X, S - I) with S=5 substituted should give Digit(X, 5-I)
        app = Digit(X, S - I)
        result = _substitute_in_predicate_application(app, S, 5)

        # The second argument should now be 5 - I (still symbolic in I)
        assert result.application_args[0] == X
        assert result.application_args[1] == 5 - I

    def test_substitute_both_variables_in_arithmetic_arg(self):
        """Substituting both variables in an arithmetic arg yields a Python int."""
        import sympy as sp
        from pysignet.logic.expansion import _substitute_in_predicate_application

        S, I = Variable("S I")
        Digit = Symbol("Digit")
        X = Variable("X")

        # Digit(X, S - I): first substitute S=5, then I=3
        app = Digit(X, S - I)
        after_s = _substitute_in_predicate_application(app, S, 5)
        after_i = _substitute_in_predicate_application(after_s, I, 3)

        # Result should be Digit(X, 2) where 2 is a plain Python int
        assert after_i.application_args[1] == 2
        assert isinstance(after_i.application_args[1], int)


class TestArithmeticPredicateArguments:
    """Tests for predicate arguments that are arithmetic expressions of quantified vars."""

    def test_forall_exists_arithmetic_arg_expansion(self):
        """ForAll-Exists with arithmetic argument S-I expands to concrete integers."""
        S, I = Variable("S I")
        Digit = Symbol("Digit")
        X1, X2 = Variable("X1 X2")

        # Inner: Exists(I, [0,1,2], Digit(X2, S-I))
        # With S=2 this should give Digit(X2,2) v Digit(X2,1) v Digit(X2,0)
        exists = Exists(I, [0, 1, 2], Digit(X2, S - I))
        outer = ForAll(S, [2], exists)
        expanded = expand_quantifier(outer)

        # S=2 only: Digit(X2,2) v Digit(X2,1) v Digit(X2,0)
        expected = sp.Or(Digit(X2, 2), Digit(X2, 1), Digit(X2, 0))
        assert expanded.equals(expected)

    def test_arithmetic_arg_becomes_python_int_not_sympy_integer(self):
        """After full substitution the argument is a Python int, not a SymPy Integer."""
        S, I = Variable("S I")
        Digit = Symbol("Digit")
        X = Variable("X")

        exists = Exists(I, [3], Digit(X, S - I))
        outer = ForAll(S, [5], exists)
        expanded = expand_quantifier(outer)

        # Expanded to Digit(X, 2)
        arg = expanded.application_args[1]
        assert arg == 2
        assert isinstance(arg, int), (
            f"Expected Python int, got {type(arg)}: {arg!r}"
        )

    def test_negative_arithmetic_arg_preserved(self):
        """Out-of-range arithmetic args (negative) are preserved as Python int."""
        S, I = Variable("S I")
        Digit = Symbol("Digit")
        X = Variable("X")

        # S=0, I=5 => S-I = -5
        exists = Exists(I, [5], Digit(X, S - I))
        outer = ForAll(S, [0], exists)
        expanded = expand_quantifier(outer)

        arg = expanded.application_args[1]
        assert arg == -5
        assert isinstance(arg, int)

    def test_addition_constraint_pattern(self):
        """Full ForAll-Implies-Exists pattern from the MNIST addition notebook."""
        import torch
        from pysignet import logic_to_loss
        from pysignet.logic.quantifier import ForAll, Exists

        Sum = Symbol("Sum")
        Digit = Symbol("Digit")
        X1, X2, S_actual, S, I = Variable("X1 X2 S_actual S I")

        expr = ForAll(
            S, range(19),
            sp.Implies(
                Sum(S_actual, S),
                Exists(I, range(10), sp.And(Digit(X1, I), Digit(X2, S - I))),
            ),
        )

        received_args = []

        def digit_fn(x, digit_idx):
            received_args.append(digit_idx)
            if not (0 <= digit_idx <= 9):
                return torch.zeros(x.shape[0])
            return torch.full((x.shape[0],), 0.5)

        predicates = {
            "Sum": lambda s_actual, s: (s_actual == s).float(),
            "Digit": digit_fn,
        }

        loss_fn = logic_to_loss(expr, predicates)
        x = torch.randn(4, 10)
        sums = torch.tensor([3, 5, 7, 9])

        # Must not raise TypeError about SymPy relational truth values
        loss = loss_fn.loss(X1=x, X2=x, S_actual=sums)

        assert loss.shape == ()
        assert not torch.isnan(loss)
        # All digit_idx args received by the predicate must be plain Python ints
        for idx in received_args:
            assert isinstance(idx, int), (
                f"Expected Python int, got {type(idx)}: {idx!r}"
            )
