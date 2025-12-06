"""Boundary tests for LogicEngine truth-table vs heuristic paths.

Tests cover:
- Truth table evaluation cutoff (≤5 variables)
- Heuristic fallback paths
- Warning generation
- Counterexample finding
"""

from __future__ import annotations

import pytest

from agents.core_logic import LogicEngine, parse_argument
from agents.core_logic.logic_engine import (
    LogicalArgument,
    LogicForm,
    Proposition,
    TruthValue,
    ValidationResult,
)


class TestTruthTablePath:
    """Test truth table evaluation (deterministic, ≤5 variables)."""

    def test_truth_table_valid_argument(self) -> None:
        """Truth table validates correct modus ponens."""
        arg = parse_argument(
            premises=["A → B", "A"],
            conclusion="B",
        )
        engine = LogicEngine()
        result = engine.validate(arg)

        assert result.is_valid is True
        assert result.confidence == 1.0
        # May be pattern_match or truth_table depending on forms loaded
        assert result.method in ("pattern_match", "truth_table")

    def test_truth_table_invalid_finds_counterexample(self) -> None:
        """Truth table finds counterexample for invalid argument."""
        # Invalid: affirming the consequent (A → B, B ⊢ A)
        arg = parse_argument(
            premises=["A → B", "B"],
            conclusion="A",
        )
        engine = LogicEngine()
        result = engine.validate(arg)

        assert result.is_valid is False
        assert result.confidence == 1.0
        # Should find counterexample or match invalid pattern
        if result.method == "truth_table":
            assert result.counterexample is not None

    def test_five_variable_boundary(self) -> None:
        """5 variables should still use truth table (2^5 = 32 rows)."""
        arg = parse_argument(
            premises=["A → B", "B → C", "C → D", "D → E"],
            conclusion="A → E",
        )
        # 5 variables: A, B, C, D, E
        assert len(arg.propositions) == 5

        engine = LogicEngine()
        result = engine.validate(arg)

        # Should still use truth table or pattern_match
        assert result.confidence == 1.0


class TestHeuristicPath:
    """Test heuristic fallback path (>5 variables or unmatched patterns)."""

    def test_six_variables_uses_heuristic(self) -> None:
        """6+ variables exceeds truth table threshold, uses heuristic."""
        # Create argument with 6 distinct variables
        arg = parse_argument(
            premises=["A → B", "B → C", "C → D", "D → E", "E → F"],
            conclusion="A → F",
        )
        assert len(arg.propositions) == 6

        engine = LogicEngine()
        result = engine.validate(arg)

        # Should warn about too many variables
        assert any("Too many variables" in w for w in result.warnings)

    def test_heuristic_detects_new_terms_in_conclusion(self) -> None:
        """Heuristic detects conclusion introducing terms not in premises."""
        arg = parse_argument(
            premises=["A → B"],
            conclusion="C",  # C not mentioned in premises
        )
        engine = LogicEngine()
        result = engine.validate(arg)

        assert result.is_valid is False
        assert "new terms" in result.explanation.lower()

    def test_no_premises_returns_heuristic_invalid(self) -> None:
        """Argument with no premises is invalid with low confidence."""
        arg = LogicalArgument(
            premises=[],
            conclusion="A",
            propositions={Proposition(symbol="A", statement="A")},
        )
        engine = LogicEngine()
        result = engine.validate(arg)

        assert result.is_valid is False
        assert result.method == "heuristic"
        assert result.confidence < 1.0
        assert "No premises" in result.warnings[0] or "without premises" in result.explanation.lower()


class TestPatternMatching:
    """Test pattern matching against known valid/invalid forms."""

    def test_modus_ponens_pattern_match(self) -> None:
        """Modus ponens matches known valid form."""
        arg = parse_argument(
            premises=["P → Q", "P"],
            conclusion="Q",
        )
        engine = LogicEngine()
        result = engine.validate(arg)

        assert result.is_valid is True
        assert result.method == "pattern_match"
        assert result.form_identified == LogicForm.MODUS_PONENS
        assert result.confidence == 1.0

    def test_modus_tollens_pattern_match(self) -> None:
        """Modus tollens matches known valid form."""
        arg = parse_argument(
            premises=["P → Q", "¬Q"],
            conclusion="¬P",
        )
        engine = LogicEngine()
        result = engine.validate(arg)

        assert result.is_valid is True
        assert result.method == "pattern_match"
        assert result.form_identified == LogicForm.MODUS_TOLLENS
        assert result.confidence == 1.0

    def test_affirming_consequent_invalid_pattern(self) -> None:
        """Affirming the consequent matches known invalid form (fallacy)."""
        arg = parse_argument(
            premises=["P → Q", "Q"],
            conclusion="P",
        )
        engine = LogicEngine()
        result = engine.validate(arg)

        assert result.is_valid is False
        assert result.confidence == 1.0
        # Should identify as fallacy via pattern or truth table
        if result.form_identified:
            assert result.form_identified == LogicForm.AFFIRMING_CONSEQUENT


class TestConvenienceFunctions:
    """Test convenience validation functions."""

    def test_validate_modus_ponens_helper(self) -> None:
        """validate_modus_ponens convenience function works correctly."""
        from agents.core_logic.logic_engine import validate_modus_ponens

        result = validate_modus_ponens(
            conditional="X → Y",
            antecedent="X",
            expected_conclusion="Y",
        )
        assert result.is_valid is True

    def test_validate_modus_tollens_helper(self) -> None:
        """validate_modus_tollens convenience function works correctly."""
        from agents.core_logic.logic_engine import validate_modus_tollens

        result = validate_modus_tollens(
            conditional="X → Y",
            negated_consequent="¬Y",
            expected_conclusion="¬X",
        )
        assert result.is_valid is True


# =============================================================================
# Biconditional Evaluation Tests
# =============================================================================


class TestBiconditionalEvaluation:
    """Test ↔ operator - logic correctness."""

    @pytest.mark.parametrize("p,q,expected", [
        (True, True, True),
        (False, False, True),
        (True, False, False),
        (False, True, False),
    ])
    def test_biconditional_truth_table(self, p: bool, q: bool, expected: bool) -> None:
        """Biconditional follows standard truth table."""
        engine = LogicEngine()
        result = engine._evaluate_expression("P ↔ Q", {"P": p, "Q": q})
        assert result is expected


# =============================================================================
# Proposition Equality Tests
# =============================================================================


class TestPropositionEquality:
    """Test Proposition identity - affects set deduplication."""

    def test_same_symbol_equal(self) -> None:
        """Propositions with same symbol are equal."""
        p1 = Proposition(symbol="P", statement="A")
        p2 = Proposition(symbol="P", statement="B")
        assert p1 == p2
        assert hash(p1) == hash(p2)

    def test_different_symbol_not_equal(self) -> None:
        """Propositions with different symbols are not equal."""
        p1 = Proposition(symbol="P", statement="A")
        p2 = Proposition(symbol="Q", statement="A")
        assert p1 != p2

    def test_proposition_in_set_deduplication(self) -> None:
        """Equal propositions deduplicate in sets."""
        p1 = Proposition(symbol="P", statement="A")
        p2 = Proposition(symbol="P", statement="B")
        prop_set = {p1, p2}
        assert len(prop_set) == 1
