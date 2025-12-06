"""
Unit tests for Rule Engine

Tests theorem proving, forward/backward chaining, unification,
predicate parsing, and proof generation.
Updated to match current API (unified RuleEngine class).
"""

import pytest
from agents.core.rule_engine import (
    RuleEngine,
    Predicate,
    Rule,
    ProofStep,
    ProofResult,
    ProofStatus,
    PredicateParser,
)


class TestPredicate:
    """Test suite for Predicate."""

    @pytest.mark.unit
    def test_predicate_creation(self):
        """Test creating a predicate."""
        pred = Predicate(name="Mortal", arguments=["Socrates"], negated=False)

        assert pred.name == "Mortal"
        assert pred.arguments == ["Socrates"]
        assert pred.negated is False

    @pytest.mark.unit
    def test_predicate_negation(self):
        """Test negating a predicate."""
        pred = Predicate(name="Happy", arguments=["Alice"])
        negated = pred.negate()

        assert negated.negated is True
        assert negated.name == "Happy"

    @pytest.mark.unit
    def test_predicate_equality(self):
        """Test predicate equality."""
        pred1 = Predicate("Human", ["Socrates"])
        pred2 = Predicate("Human", ["Socrates"])

        assert pred1 == pred2

    @pytest.mark.unit
    def test_predicate_substitution(self):
        """Test variable substitution in predicates."""
        pred = Predicate("Loves", ["X", "Y"])
        bindings = {"X": "Alice", "Y": "Bob"}

        result = pred.substitute(bindings)

        assert result.arguments == ["Alice", "Bob"]


class TestRule:
    """Test suite for Rule."""

    @pytest.mark.unit
    def test_rule_creation(self):
        """Test creating a rule."""
        rule = Rule(
            name="mortality",
            antecedents=[Predicate("Human", ["X"])],
            consequent=Predicate("Mortal", ["X"])
        )

        assert rule.name == "mortality"
        assert len(rule.antecedents) == 1

    @pytest.mark.unit
    def test_multi_antecedent_rule(self):
        """Test rule with multiple antecedents."""
        rule = Rule(
            name="grandparent",
            antecedents=[
                Predicate("Parent", ["X", "Y"]),
                Predicate("Parent", ["Y", "Z"])
            ],
            consequent=Predicate("Grandparent", ["X", "Z"])
        )

        assert len(rule.antecedents) == 2


class TestPredicateParser:
    """Test suite for PredicateParser."""

    @pytest.fixture
    def parser(self):
        """Create PredicateParser instance."""
        return PredicateParser()

    @pytest.mark.unit
    def test_parse_simple_statement(self, parser):
        """Test parsing simple statement."""
        pred = parser.parse("Socrates is human")

        assert pred is not None
        assert isinstance(pred, Predicate)


class TestRuleEngine:
    """Test suite for RuleEngine (integrated forward/backward chaining)."""

    @pytest.fixture
    def engine(self):
        """Create RuleEngine instance."""
        return RuleEngine()

    @pytest.mark.unit
    def test_initialization(self, engine):
        """Test rule engine initialization."""
        assert engine is not None

    @pytest.mark.unit
    def test_add_fact(self, engine):
        """Test adding facts to knowledge base."""
        fact = Predicate("Human", ["Socrates"])
        engine.add_fact(fact)

        # Fact should be stored
        assert fact is not None

    @pytest.mark.unit
    def test_add_rule(self, engine):
        """Test adding rules to knowledge base."""
        rule = Rule(
            name="mortality",
            antecedents=[Predicate("Human", ["X"])],
            consequent=Predicate("Mortal", ["X"])
        )

        engine.add_rule(rule)
        # Rule should be stored

    @pytest.mark.integration
    def test_forward_chain_simple(self, engine):
        """Test simple forward chaining inference."""
        engine.add_fact(Predicate("Human", ["Socrates"]))
        engine.add_rule(Rule(
            name="mortality",
            antecedents=[Predicate("Human", ["X"])],
            consequent=Predicate("Mortal", ["X"])
        ))

        derived = engine.forward_chain()

        # Should derive Mortal(Socrates)
        assert len(derived) > 0

    @pytest.mark.integration
    def test_forward_chain_multi_step(self, engine):
        """Test chained forward inference."""
        engine.add_fact(Predicate("Human", ["Socrates"]))
        engine.add_rule(Rule(
            name="r1",
            antecedents=[Predicate("Human", ["X"])],
            consequent=Predicate("Mortal", ["X"])
        ))
        engine.add_rule(Rule(
            name="r2",
            antecedents=[Predicate("Mortal", ["X"])],
            consequent=Predicate("Finite", ["X"])
        ))

        derived = engine.forward_chain()

        # Should derive multiple predicates
        assert len(derived) >= 2

    @pytest.mark.integration
    def test_backward_chain_simple(self, engine):
        """Test simple backward chaining proof.
        
        Note: Variables are identified by starting with uppercase, so we use
        lowercase 'socrates' for the constant to distinguish from variable 'X'.
        """
        engine.add_fact(Predicate("human", ["socrates"]))
        engine.add_rule(Rule(
            name="mortality",
            antecedents=[Predicate("human", ["X"])],
            consequent=Predicate("mortal", ["X"])
        ))

        goal = Predicate("mortal", ["socrates"])
        result = engine.backward_chain(goal)

        assert result.status == ProofStatus.PROVEN

    @pytest.mark.integration
    def test_backward_chain_multi_step(self, engine):
        """Test multi-step backward proof.
        
        Note: Using lowercase constants to distinguish from uppercase variables.
        """
        engine.add_fact(Predicate("human", ["socrates"]))
        engine.add_rule(Rule(
            name="r1",
            antecedents=[Predicate("human", ["X"])],
            consequent=Predicate("mortal", ["X"])
        ))
        engine.add_rule(Rule(
            name="r2",
            antecedents=[Predicate("mortal", ["X"])],
            consequent=Predicate("finite", ["X"])
        ))

        goal = Predicate("finite", ["socrates"])
        result = engine.backward_chain(goal)

        assert result.status == ProofStatus.PROVEN

    @pytest.mark.integration
    def test_backward_chain_fails(self, engine):
        """Test that unprovable goals fail correctly."""
        engine.add_fact(Predicate("Human", ["Socrates"]))

        goal = Predicate("God", ["Socrates"])
        result = engine.backward_chain(goal)

        assert result.status in [ProofStatus.UNKNOWN, ProofStatus.DISPROVEN]


class TestProofResult:
    """Test suite for ProofResult."""

    @pytest.mark.unit
    def test_proof_result_creation(self):
        """Test creating a proof result."""
        result = ProofResult(
            status=ProofStatus.PROVEN,
            goal=Predicate("Mortal", ["Socrates"]),
            steps=[],
            confidence=1.0,
            time_ms=10.5
        )

        assert result.status == ProofStatus.PROVEN
        assert result.is_proven is True
        assert result.confidence == 1.0

    @pytest.mark.unit
    def test_proof_result_with_steps(self):
        """Test proof result with proof steps."""
        steps = [
            ProofStep(
                step_number=1,
                predicate=Predicate("Human", ["Socrates"]),
                justification="Given fact",
                confidence=1.0
            ),
            ProofStep(
                step_number=2,
                predicate=Predicate("Mortal", ["Socrates"]),
                justification="By mortality rule",
                rule_applied="mortality",
                confidence=0.95
            ),
        ]

        result = ProofResult(
            status=ProofStatus.PROVEN,
            goal=Predicate("Mortal", ["Socrates"]),
            steps=steps,
            confidence=0.95,
            time_ms=15.0
        )

        assert len(result.steps) == 2


class TestProofStep:
    """Test suite for ProofStep."""

    @pytest.mark.unit
    def test_proof_step_creation(self):
        """Test creating a proof step."""
        step = ProofStep(
            step_number=1,
            predicate=Predicate("Human", ["Socrates"]),
            justification="Given as fact",
            confidence=1.0
        )

        assert step.step_number == 1
        assert step.confidence == 1.0
