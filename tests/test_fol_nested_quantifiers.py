"""Comprehensive tests for nested quantifiers through compilation.

This module tests that nested ForAll and Exists quantifiers work correctly
end-to-end through the compilation and evaluation pipeline, with proper
variable scoping and gradient flow.
"""

import torch
import torch.nn as nn
import sympy as sp

from pysignet import Symbol, compile_logic
from pysignet.logic import Variable, ForAll, Exists


class TestForAllForAllNesting:
    """Tests for ForAll nested within ForAll."""

    def test_forall_forall_basic(self):
        """Basic ForAll-ForAll nesting compiles and evaluates."""
        X, Y, Z = Variable("X Y Z")
        P = Symbol("P")

        # ForAll(Y, [0, 1], ForAll(Z, [2, 3], P(X, Y, Z)))
        # X is free (auto-batched), Y and Z are quantified over domains
        inner = ForAll(Z, [2, 3], P(X, Y, Z))
        expr = ForAll(Y, [0, 1], inner)

        # Ternary predicate: 2 free vars (Y, Z) + batch var (X)
        # Model outputs: 2*2 = 4 combinations for (Y,Z) pairs
        model = nn.Sequential(nn.Linear(5, 4), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"P": model})

        # Evaluate with batch
        x = torch.randn(3, 5)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        # Result should be (3,) - per-batch satisfaction
        assert result.shape == (3,)
        assert torch.all((result >= 0) & (result <= 1))

    def test_forall_forall_different_domains(self):
        """ForAll-ForAll with different domain sizes."""
        X, Y, Z = Variable("X Y Z")
        R = Symbol("R")

        # Outer domain: [0, 1, 2] (3 elements)
        # Inner domain: [5, 10] (2 elements)
        # Total: 3*2 = 6 combinations
        inner = ForAll(Z, [5, 10], R(X, Y, Z))
        expr = ForAll(Y, [0, 1, 2], inner)

        # Model outputs 6 combinations (3*2)
        model = nn.Sequential(nn.Linear(4, 6), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"R": model})

        x = torch.randn(2, 4)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        assert result.shape == (2,)

    def test_forall_forall_with_complex_body(self):
        """ForAll-ForAll with complex expression in body."""
        X, Y, Z = Variable("X Y Z")
        P, Q = Symbol("P Q")

        # ForAll(Y, [0, 1], ForAll(Z, [2, 3], P(X, Y, Z) → Q(X, Y)))
        inner_body = sp.Implies(P(X, Y, Z), Q(X, Y))
        inner = ForAll(Z, [2, 3], inner_body)
        expr = ForAll(Y, [0, 1], inner)

        # P has 2 free vars (Y, Z): 2*2 = 4 outputs
        p_model = nn.Sequential(nn.Linear(5, 4), nn.Softmax(dim=-1))

        # Q has 1 free var (Y): 2 outputs
        q_model = nn.Sequential(nn.Linear(5, 2), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"P": p_model, "Q": q_model})

        x = torch.randn(3, 5)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        assert result.shape == (3,)

    def test_forall_forall_single_element_domains(self):
        """ForAll-ForAll with single-element domains."""
        X, Y, Z = Variable("X Y Z")
        P = Symbol("P")

        # Both quantifiers have single-element domains
        inner = ForAll(Z, [5], P(X, Y, Z))
        expr = ForAll(Y, [3], inner)

        # Only 1*1 = 1 combination
        model = nn.Sequential(nn.Linear(3, 1), nn.Sigmoid())

        def model_func(inputs, y, z):
            return model(inputs).squeeze(-1) + y + z

        compiled = compile_logic(expr, {"P": model_func})

        x = torch.randn(4, 3)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        assert result.shape == (4,)


class TestExistsExistsNesting:
    """Tests for Exists nested within Exists."""

    def test_exists_exists_basic(self):
        """Basic Exists-Exists nesting compiles and evaluates."""
        X, Y, Z = Variable("X Y Z")
        P = Symbol("P")

        # Exists(Y, [0, 1], Exists(Z, [2, 3], P(X, Y, Z)))
        inner = Exists(Z, [2, 3], P(X, Y, Z))
        expr = Exists(Y, [0, 1], inner)

        # 2 free vars (Y, Z): 2*2 = 4 outputs
        model = nn.Sequential(nn.Linear(5, 4), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"P": model})

        x = torch.randn(3, 5)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        # Existential quantification over batch
        assert result.shape == (3,)
        assert torch.all((result >= 0) & (result <= 1))

    def test_exists_exists_different_domains(self):
        """Exists-Exists with different domain sizes."""
        X, Y, Z = Variable("X Y Z")
        P = Symbol("P")

        # Outer: [0, 1, 2] (3), Inner: [10, 20] (2) = 6 combinations
        inner = Exists(Z, [10, 20], P(X, Y, Z))
        expr = Exists(Y, [0, 1, 2], inner)

        model = nn.Sequential(nn.Linear(4, 6), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"P": model})

        x = torch.randn(2, 4)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        assert result.shape == (2,)

    def test_exists_exists_with_conjunction(self):
        """Exists-Exists with conjunction in body."""
        X, Y, Z = Variable("X Y Z")
        P, Q = Symbol("P Q")

        # Exists(Y, [0, 1], Exists(Z, [0, 1], P(X, Y, Z) ∧ Q(X, Z)))
        # Using domain [0, 1] for both to match output indices
        inner_body = sp.And(P(X, Y, Z), Q(X, Z))
        inner = Exists(Z, [0, 1], inner_body)
        expr = Exists(Y, [0, 1], inner)

        # P: 2 free vars (Y, Z) = 2*2 = 4 outputs
        p_model = nn.Sequential(nn.Linear(5, 4), nn.Softmax(dim=-1))

        # Q: 1 free var (Z) = 2 outputs
        q_model = nn.Sequential(nn.Linear(5, 2), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"P": p_model, "Q": q_model})

        x = torch.randn(3, 5)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        assert result.shape == (3,)


class TestMixedQuantifierNesting:
    """Tests for ForAll and Exists mixed nesting."""

    def test_forall_exists_nesting(self):
        """ForAll containing Exists."""
        X, Y, Z = Variable("X Y Z")
        P = Symbol("P")

        # ForAll(Y, [0, 1], Exists(Z, [2, 3], P(X, Y, Z)))
        # "For all Y, there exists Z such that P(X, Y, Z)"
        inner = Exists(Z, [2, 3], P(X, Y, Z))
        expr = ForAll(Y, [0, 1], inner)

        model = nn.Sequential(nn.Linear(5, 4), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"P": model})

        x = torch.randn(3, 5)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        assert result.shape == (3,)
        assert torch.all((result >= 0) & (result <= 1))

    def test_exists_forall_nesting(self):
        """Exists containing ForAll."""
        X, Y, Z = Variable("X Y Z")
        P = Symbol("P")

        # Exists(Y, [0, 1], ForAll(Z, [2, 3], P(X, Y, Z)))
        # "There exists Y such that for all Z, P(X, Y, Z)"
        inner = ForAll(Z, [2, 3], P(X, Y, Z))
        expr = Exists(Y, [0, 1], inner)

        model = nn.Sequential(nn.Linear(5, 4), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"P": model})

        x = torch.randn(3, 5)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        assert result.shape == (3,)

    def test_forall_exists_with_implication(self):
        """ForAll-Exists with implication."""
        X, Y, Z = Variable("X Y Z")
        P, Q = Symbol("P Q")

        # ForAll(Y, [0, 1], Exists(Z, [0, 1], P(X, Y) → Q(X, Z)))
        # Using domain [0, 1] to match output indices
        inner_body = sp.Implies(P(X, Y), Q(X, Z))
        inner = Exists(Z, [0, 1], inner_body)
        expr = ForAll(Y, [0, 1], inner)

        # P: 1 free var (Y) = 2 outputs
        p_model = nn.Sequential(nn.Linear(4, 2), nn.Softmax(dim=-1))

        # Q: 1 free var (Z) = 2 outputs
        q_model = nn.Sequential(nn.Linear(4, 2), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"P": p_model, "Q": q_model})

        x = torch.randn(2, 4)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        assert result.shape == (2,)

    def test_exists_forall_with_equivalence(self):
        """Exists-ForAll with equivalence."""
        X, Y, Z = Variable("X Y Z")
        P, Q = Symbol("P Q")

        # Exists(Y, [0, 1], ForAll(Z, [2], P(X, Y) ↔ Q(X, Z)))
        inner_body = sp.Equivalent(P(X, Y), Q(X, Z))
        inner = ForAll(Z, [2], inner_body)
        expr = Exists(Y, [0, 1], inner)

        p_model = nn.Sequential(nn.Linear(3, 2), nn.Softmax(dim=-1))
        q_model = nn.Sequential(nn.Linear(3, 1), nn.Sigmoid())

        def q_func(inputs, z):
            return q_model(inputs).squeeze(-1) * z

        compiled = compile_logic(expr, {"P": p_model, "Q": q_func})

        x = torch.randn(2, 3)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        assert result.shape == (2,)


class TestTripleNesting:
    """Tests for triple-nested quantifiers."""

    def test_forall_forall_forall(self):
        """Triple ForAll nesting."""
        W, X, Y, Z = Variable("W X Y Z")
        P = Symbol("P")

        # ForAll(X, [0, 1], ForAll(Y, [2], ForAll(Z, [3, 4], P(W, X, Y, Z))))
        innermost = ForAll(Z, [3, 4], P(W, X, Y, Z))
        middle = ForAll(Y, [2], innermost)
        expr = ForAll(X, [0, 1], middle)

        # 3 free vars (X, Y, Z): 2*1*2 = 4 outputs
        model = nn.Sequential(nn.Linear(5, 4), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"P": model})

        w = torch.randn(3, 5)
        # Use quantify='none' to get per-batch results
        result = compiled(w, quantify='none')

        assert result.shape == (3,)

    def test_exists_exists_exists(self):
        """Triple Exists nesting."""
        W, X, Y, Z = Variable("W X Y Z")
        P = Symbol("P")

        # Exists(X, [0, 1], Exists(Y, [2], Exists(Z, [3, 4], P(W, X, Y, Z))))
        innermost = Exists(Z, [3, 4], P(W, X, Y, Z))
        middle = Exists(Y, [2], innermost)
        expr = Exists(X, [0, 1], middle)

        # 3 free vars: 2*1*2 = 4 outputs
        model = nn.Sequential(nn.Linear(5, 4), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"P": model})

        w = torch.randn(3, 5)
        # Use quantify='none' to get per-batch results
        result = compiled(w, quantify='none')

        assert result.shape == (3,)

    def test_forall_exists_forall(self):
        """Alternating ForAll-Exists-ForAll."""
        W, X, Y, Z = Variable("W X Y Z")
        P = Symbol("P")

        # ForAll(X, [0, 1], Exists(Y, [2, 3], ForAll(Z, [4], P(W, X, Y, Z))))
        innermost = ForAll(Z, [4], P(W, X, Y, Z))
        middle = Exists(Y, [2, 3], innermost)
        expr = ForAll(X, [0, 1], middle)

        # 3 free vars: 2*2*1 = 4 outputs
        model = nn.Sequential(nn.Linear(6, 4), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"P": model})

        w = torch.randn(2, 6)
        # Use quantify='none' to get per-batch results
        result = compiled(w, quantify='none')

        assert result.shape == (2,)


class TestVariableScoping:
    """Tests for proper variable scoping in nested quantifiers."""

    def test_inner_variable_shadows_correctly(self):
        """Inner quantifier variable has correct scope."""
        X, Y, Z = Variable("X Y Z")
        P = Symbol("P")

        # ForAll(Y, [0, 1], ForAll(Z, [0, 1], P(X, Y, Z)))
        # Y's scope is the outer ForAll
        # Z's scope is the inner ForAll
        # X is free (auto-batched)
        inner = ForAll(Z, [0, 1], P(X, Y, Z))
        expr = ForAll(Y, [0, 1], inner)

        # P: 2 free vars (Y, Z) = 2*2 = 4 outputs
        p_model = nn.Sequential(nn.Linear(4, 4), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"P": p_model})

        x = torch.randn(3, 4)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        assert result.shape == (3,)

    def test_free_variables_in_nested_context(self):
        """Free variables remain free in nested quantifiers."""
        X, Y, Z = Variable("X Y Z")
        P = Symbol("P")

        # X is free (auto-batched)
        # Y and Z are quantified
        # ForAll(Y, [0, 1], Exists(Z, [0, 1], P(X, Y, Z)))
        inner = Exists(Z, [0, 1], P(X, Y, Z))
        expr = ForAll(Y, [0, 1], inner)

        # P: 2 free vars (Y, Z) = 2*2 = 4 outputs
        model = nn.Sequential(nn.Linear(5, 4), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"P": model})

        # X is the free batch variable
        x = torch.randn(3, 5)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        assert result.shape == (3,)

    def test_quantified_variable_not_free(self):
        """Quantified variables are not free."""
        X, Y = Variable("X Y")
        P = Symbol("P")

        # Only X is free (Y is quantified)
        expr = ForAll(Y, [0, 1, 2], P(X, Y))

        # P has 1 free var (Y): 3 outputs
        model = nn.Sequential(nn.Linear(5, 3), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"P": model})

        # Should accept just X
        x = torch.randn(4, 5)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        assert result.shape == (4,)

    def test_all_variables_quantified_minimal_batch(self):
        """Expression with all domain variables quantified."""
        X, Y, Z = Variable("X Y Z")
        P = Symbol("P")

        # X is free (batch), Y and Z are quantified over domains
        inner = ForAll(Z, [0, 1], P(X, Y, Z))
        expr = ForAll(Y, [0, 1], inner)

        # 2 free vars (Y, Z): 2*2 = 4 outputs
        model = nn.Sequential(nn.Linear(3, 4), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"P": model})

        # X is the batch variable
        x = torch.randn(2, 3)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        # Batch dimension preserved
        assert result.shape == (2,)


class TestGradientFlow:
    """Tests for gradient flow through nested quantifiers."""

    def test_gradients_through_forall_forall(self):
        """Gradients flow through nested ForAll."""
        X, Y, Z = Variable("X Y Z")
        P = Symbol("P")

        inner = ForAll(Z, [2, 3], P(X, Y, Z))
        expr = ForAll(Y, [0, 1], inner)

        model = nn.Sequential(nn.Linear(5, 4), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"P": model})

        x = torch.randn(3, 5)
        # Default quantify='forall' returns scalar, suitable for loss
        loss = compiled.loss(x)

        loss.backward()

        # Check all parameters have gradients
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_gradients_through_exists_exists(self):
        """Gradients flow through nested Exists."""
        X, Y, Z = Variable("X Y Z")
        P = Symbol("P")

        inner = Exists(Z, [2, 3], P(X, Y, Z))
        expr = Exists(Y, [0, 1], inner)

        model = nn.Sequential(nn.Linear(4, 4), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"P": model})

        x = torch.randn(3, 4)
        # Default quantify='forall' returns scalar, suitable for loss
        loss = compiled.loss(x)

        loss.backward()

        for param in model.parameters():
            assert param.grad is not None

    def test_gradients_through_mixed_nesting(self):
        """Gradients flow through ForAll-Exists nesting."""
        X, Y, Z = Variable("X Y Z")
        P = Symbol("P")

        inner = Exists(Z, [2, 3], P(X, Y, Z))
        expr = ForAll(Y, [0, 1], inner)

        model = nn.Sequential(nn.Linear(5, 4), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"P": model})

        x = torch.randn(3, 5)
        # Default quantify='forall' returns scalar, suitable for loss
        loss = compiled.loss(x)

        loss.backward()

        for param in model.parameters():
            assert param.grad is not None

    def test_gradients_through_triple_nesting(self):
        """Gradients flow through triple-nested quantifiers."""
        W, X, Y, Z = Variable("W X Y Z")
        P = Symbol("P")

        innermost = ForAll(Z, [3, 4], P(W, X, Y, Z))
        middle = Exists(Y, [2], innermost)
        expr = ForAll(X, [0, 1], middle)

        model = nn.Sequential(nn.Linear(5, 4), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"P": model})

        w = torch.randn(3, 5)
        # Default quantify='forall' returns scalar, suitable for loss
        loss = compiled.loss(w)

        loss.backward()

        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


class TestRealWorldPatterns:
    """Tests for realistic nested quantifier patterns."""

    def test_all_pairs_constraint(self):
        """Constraint over all pairs of values."""
        X, Y, Z = Variable("X Y Z")
        Similar = Symbol("Similar")

        # ForAll(Y, [0,1,2], ForAll(Z, [0,1,2], Similar(X, Y, Z)))
        # "For all pairs (Y, Z), X is similar to the pair"
        inner = ForAll(Z, [0, 1, 2], Similar(X, Y, Z))
        expr = ForAll(Y, [0, 1, 2], inner)

        # 2 free vars (Y, Z): 3*3 = 9 outputs
        model = nn.Sequential(nn.Linear(10, 9), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"Similar": model})

        x = torch.randn(4, 10)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        assert result.shape == (4,)

    def test_mutual_exclusion_constraint(self):
        """At most one condition holds."""
        X, Y = Variable("X Y")
        Digit, Prime = Symbol("Digit Prime")

        # Exists(Y, [2,3,5,7], Digit(X, Y)) → ¬Prime(X)
        # "If X is a prime digit, then it's not composite-prime"
        exists_expr = Exists(Y, [2, 3, 5, 7], Digit(X, Y))
        expr = sp.Implies(exists_expr, sp.Not(Prime(X)))

        digit_model = nn.Sequential(nn.Linear(8, 10), nn.Softmax(dim=-1))
        prime_model = nn.Sequential(nn.Linear(8, 1), nn.Sigmoid())

        def prime_func(inputs):
            return prime_model(inputs).squeeze(-1)

        compiled = compile_logic(expr, {"Digit": digit_model, "Prime": prime_func})

        x = torch.randn(3, 8)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        assert result.shape == (3,)

    def test_hierarchical_classification(self):
        """Hierarchical constraints with nested quantifiers."""
        X, Coarse, Fine = Variable("X Coarse Fine")
        Class = Symbol("Class")

        # ForAll(Coarse, [0,1], Exists(Fine, [0,1,2], Class(X, Coarse, Fine)))
        # "For each coarse category, X belongs to at least one fine subcategory"
        inner = Exists(Fine, [0, 1, 2], Class(X, Coarse, Fine))
        expr = ForAll(Coarse, [0, 1], inner)

        # 2 free vars (Coarse, Fine): 2*3 = 6 outputs
        model = nn.Sequential(nn.Linear(12, 6), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"Class": model})

        x = torch.randn(2, 12)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        assert result.shape == (2,)


class TestEdgeCases:
    """Tests for edge cases in nested quantifiers."""

    def test_empty_inner_domain(self):
        """Nested quantifier with empty inner domain."""
        X, Y, Z = Variable("X Y Z")
        P = Symbol("P")

        # ForAll(Y, [0, 1], ForAll(Z, [], P(X, Y, Z)))
        # Inner ForAll over empty domain = true
        # Outer becomes: ForAll(Y, [0, 1], true) = true
        inner = ForAll(Z, [], P(X, Y, Z))
        expr = ForAll(Y, [0, 1], inner)

        model = nn.Sequential(nn.Linear(3, 2), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"P": model})

        x = torch.randn(2, 3)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        # Should be all 1.0 (vacuously true)
        assert result.shape == (2,)
        assert torch.allclose(result, torch.ones(2))

    def test_empty_outer_domain(self):
        """Nested quantifier with empty outer domain."""
        X, Y, Z = Variable("X Y Z")
        P = Symbol("P")

        # ForAll(Y, [], ForAll(Z, [2, 3], P(X, Y, Z)))
        # Outer ForAll over empty domain = true
        inner = ForAll(Z, [2, 3], P(X, Y, Z))
        expr = ForAll(Y, [], inner)

        model = nn.Sequential(nn.Linear(3, 2), nn.Softmax(dim=-1))

        compiled = compile_logic(expr, {"P": model})

        x = torch.randn(2, 3)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        # Should be all 1.0
        assert result.shape == (2,)
        assert torch.allclose(result, torch.ones(2))

    def test_single_element_nesting(self):
        """All quantifiers have single-element domains."""
        X, Y, Z = Variable("X Y Z")
        P = Symbol("P")

        inner = ForAll(Z, [5], P(X, Y, Z))
        expr = ForAll(Y, [3], inner)

        # Only 1 combination
        model = nn.Sequential(nn.Linear(3, 1), nn.Sigmoid())

        def model_func(inputs, y, z):
            return model(inputs).squeeze(-1) * y * z

        compiled = compile_logic(expr, {"P": model_func})

        x = torch.randn(3, 3)
        # Use quantify='none' to get per-batch results
        result = compiled(X=x, quantify='none')

        assert result.shape == (3,)
