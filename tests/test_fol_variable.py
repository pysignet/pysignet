"""Tests for FOL Variable class.

This module tests the Variable class which represents variables in
first-order logic expressions.
"""

import pytest
import sympy as sp


class TestVariableCreation:
    """Test Variable creation and basic properties."""

    def test_single_variable_creation(self):
        """Test creating a single variable."""
        from pysignet.logic import Variable

        X = Variable("X")

        # Should be a SymPy Symbol
        assert isinstance(X, sp.Symbol)

        # Name should match
        assert str(X) == "X"
        assert X.name == "X"

    def test_multiple_variables_from_string(self):
        """Test creating multiple variables from space-separated string."""
        from pysignet.logic import Variable

        X, Y, Z = Variable("X Y Z")

        # All should be SymPy Symbols
        assert isinstance(X, sp.Symbol)
        assert isinstance(Y, sp.Symbol)
        assert isinstance(Z, sp.Symbol)

        # Names should match
        assert str(X) == "X"
        assert str(Y) == "Y"
        assert str(Z) == "Z"

    def test_two_variables_from_string(self):
        """Test creating two variables."""
        from pysignet.logic import Variable

        X, Y = Variable("X Y")

        assert str(X) == "X"
        assert str(Y) == "Y"

    def test_single_variable_returns_variable_not_tuple(self):
        """Test that single variable doesn't return a tuple."""
        from pysignet.logic import Variable

        X = Variable("X")

        # Should be a Variable, not a tuple
        assert isinstance(X, sp.Symbol)
        assert not isinstance(X, tuple)

    def test_variable_name_with_underscore(self):
        """Test variable names with underscores."""
        from pysignet.logic import Variable

        X_1, Y_2 = Variable("X_1 Y_2")

        assert str(X_1) == "X_1"
        assert str(Y_2) == "Y_2"


class TestVariableEquality:
    """Test Variable equality and hashing."""

    def test_variables_with_same_name_are_equal(self):
        """Test that variables with the same name are equal."""
        from pysignet.logic import Variable

        X1 = Variable("X")
        X2 = Variable("X")

        assert X1 == X2

    def test_variables_with_different_names_not_equal(self):
        """Test that variables with different names are not equal."""
        from pysignet.logic import Variable

        X = Variable("X")
        Y = Variable("Y")

        assert X != Y

    def test_variable_equality_from_same_call(self):
        """Test equality of variables from same creation call."""
        from pysignet.logic import Variable

        X1, Y1 = Variable("X Y")
        X2, Y2 = Variable("X Y")

        assert X1 == X2
        assert Y1 == Y2
        assert X1 != Y1

    def test_variable_hashing(self):
        """Test that variables can be hashed (for use in sets/dicts)."""
        from pysignet.logic import Variable

        X = Variable("X")
        Y = Variable("Y")

        # Should be hashable
        var_set = {X, Y, X}  # X appears twice
        assert len(var_set) == 2  # Should deduplicate

        # Should work as dict keys
        var_dict = {X: 1, Y: 2}
        assert var_dict[X] == 1
        assert var_dict[Y] == 2


class TestVariableSymPyIntegration:
    """Test Variable integration with SymPy."""

    def test_variable_in_sympy_expression(self):
        """Test that variables work in SymPy expressions."""
        from pysignet.logic import Variable

        X, Y = Variable("X Y")

        # Should work with SymPy logic operators
        expr = sp.And(X, Y)
        assert isinstance(expr, sp.And)
        assert X in expr.args
        assert Y in expr.args

    def test_variable_with_sympy_or(self):
        """Test variable with SymPy Or."""
        from pysignet.logic import Variable

        X, Y, Z = Variable("X Y Z")

        expr = sp.Or(X, sp.And(Y, Z))
        assert isinstance(expr, sp.Or)

    def test_variable_with_sympy_not(self):
        """Test variable with SymPy Not."""
        from pysignet.logic import Variable

        X = Variable("X")

        expr = sp.Not(X)
        assert isinstance(expr, sp.Not)

    def test_variable_substitution(self):
        """Test that variables can be substituted in SymPy expressions."""
        from pysignet.logic import Variable

        X, Y = Variable("X Y")

        expr = sp.And(X, Y)

        # Substitute X with True
        subst = expr.subs(X, True)
        # Result should be Y (since True AND Y = Y)
        assert subst == Y


class TestVariableRepr:
    """Test Variable string representation."""

    def test_variable_str(self):
        """Test str() representation."""
        from pysignet.logic import Variable

        X = Variable("X")
        assert str(X) == "X"

    def test_variable_repr(self):
        """Test repr() representation."""
        from pysignet.logic import Variable

        X = Variable("X")
        # SymPy symbols have repr same as str
        assert repr(X) == "X"


class TestVariableEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_string_raises_error(self):
        """Test that empty string raises an error."""
        from pysignet.logic import Variable

        with pytest.raises((ValueError, IndexError)):
            Variable("")

    def test_whitespace_only_raises_error(self):
        """Test that whitespace-only string raises an error."""
        from pysignet.logic import Variable

        with pytest.raises((ValueError, IndexError)):
            Variable("   ")

    def test_single_name_many_spaces(self):
        """Test single variable name with extra spaces."""
        from pysignet.logic import Variable

        X = Variable("  X  ")
        assert str(X) == "X"

    def test_multiple_names_extra_spaces(self):
        """Test multiple variable names with extra spaces."""
        from pysignet.logic import Variable

        X, Y = Variable("  X   Y  ")
        assert str(X) == "X"
        assert str(Y) == "Y"
