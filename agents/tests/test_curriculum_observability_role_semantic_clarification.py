"""
Unit tests for Curriculum, Observability, Role, Semantic Parser, and Clarification Systems

Combined test file for these core modules.
Updated to match current API.
"""

import pytest
from agents.core.curriculum_system import (
    CurriculumLearner,
    DifficultyLevel,
    EvalMetric,
    EvalExample,
    EvalResult,
    EvalReport,
    Evaluator,
    EvalHarness,
)
from agents.core.observability_system import (
    ObservabilitySystem,
    MetricsRegistry,
    Tracer,
    EventLogger,
    TraceEvent,
    Span,
    Trace,
    EventType,
    LogLevel,
)
from agents.core.role_system import (
    RoleBasedReasoner,
    RoleRegistry,
    RoleAdapter,
    RolePersona,
    ReasoningMode,
    ExpertiseLevel,
    CommunicationStyle,
)
from agents.core.semantic_parser import (
    EnhancedSemanticParser,
    SemanticFrame,
    ParseResult,
    QuantifierType,
    ModalityType,
)
from agents.core.clarification_system import (
    ClarificationManager,
    ClarifyingQuestion,
    AmbiguityDetector,
    QuestionGenerator,
    AmbiguityType,
    ClarificationPriority,
)


# ==================== Curriculum System Tests ====================

class TestCurriculumLearner:
    """Test suite for CurriculumLearner."""

    @pytest.fixture
    def learner(self):
        """Create CurriculumLearner instance with a harness."""
        harness = EvalHarness()
        return CurriculumLearner(harness=harness)

    @pytest.mark.unit
    def test_learner_creation(self, learner):
        """Test creating curriculum learner."""
        assert learner is not None


class TestDifficultyLevel:
    """Test suite for DifficultyLevel enum."""

    @pytest.mark.unit
    def test_difficulty_levels_exist(self):
        """Test difficulty levels are defined."""
        assert DifficultyLevel is not None


class TestEvaluator:
    """Test suite for Evaluator."""

    @pytest.fixture
    def evaluator(self):
        """Create Evaluator instance."""
        return Evaluator()

    @pytest.mark.unit
    def test_evaluator_creation(self, evaluator):
        """Test creating evaluator."""
        assert evaluator is not None


# ==================== Observability System Tests ====================

class TestObservabilitySystem:
    """Test suite for ObservabilitySystem."""

    @pytest.fixture
    def obs(self):
        """Create ObservabilitySystem instance."""
        return ObservabilitySystem()

    @pytest.mark.unit
    def test_system_creation(self, obs):
        """Test creating observability system."""
        assert obs is not None


class TestMetricsRegistry:
    """Test suite for MetricsRegistry."""

    @pytest.fixture
    def registry(self):
        """Create MetricsRegistry instance."""
        return MetricsRegistry()

    @pytest.mark.unit
    def test_registry_creation(self, registry):
        """Test creating metrics registry."""
        assert registry is not None


class TestTracer:
    """Test suite for Tracer."""

    @pytest.fixture
    def tracer(self):
        """Create Tracer instance."""
        return Tracer()

    @pytest.mark.unit
    def test_tracer_creation(self, tracer):
        """Test creating tracer."""
        assert tracer is not None


# ==================== Role System Tests ====================

class TestRoleBasedReasoner:
    """Test suite for RoleBasedReasoner."""

    @pytest.fixture
    def reasoner(self):
        """Create RoleBasedReasoner instance with a reasoning function."""
        def dummy_reasoning_fn(text: str, context: dict) -> dict:
            return {"result": text}
        return RoleBasedReasoner(reasoning_fn=dummy_reasoning_fn)

    @pytest.mark.unit
    def test_reasoner_creation(self, reasoner):
        """Test creating role-based reasoner."""
        assert reasoner is not None


class TestRoleRegistry:
    """Test suite for RoleRegistry."""

    @pytest.fixture
    def registry(self):
        """Create RoleRegistry instance."""
        return RoleRegistry()

    @pytest.mark.unit
    def test_registry_creation(self, registry):
        """Test creating role registry."""
        assert registry is not None


class TestRolePersona:
    """Test suite for RolePersona."""

    @pytest.mark.unit
    def test_persona_creation(self):
        """Test creating a role persona."""
        from agents.core.role_system import ReasoningMode
        
        # ReasoningMode is a dataclass, not an enum
        analytical_mode = ReasoningMode(
            name="analytical",
            description="Analytical reasoning",
            required_steps=["gather_data", "analyze", "conclude"],
            forbidden_fallacies=["hasty_generalization"],
            preferred_argument_forms=["inductive"]
        )
        
        persona = RolePersona(
            role_id="analyst",
            name="Data Analyst",
            description="Analyzes data patterns",
            domain="data_science",
            expertise_level=ExpertiseLevel.EXPERT,
            communication_style=CommunicationStyle.TECHNICAL,
            reasoning_mode=analytical_mode,
        )
        assert persona.role_id == "analyst"
        assert persona.expertise_level == ExpertiseLevel.EXPERT


# ==================== Semantic Parser Tests ====================

class TestEnhancedSemanticParser:
    """Test suite for EnhancedSemanticParser."""

    @pytest.fixture
    def parser(self):
        """Create EnhancedSemanticParser instance."""
        return EnhancedSemanticParser()

    @pytest.mark.unit
    def test_parser_creation(self, parser):
        """Test creating semantic parser."""
        assert parser is not None


class TestSemanticFrame:
    """Test suite for SemanticFrame."""

    @pytest.mark.unit
    def test_frame_creation(self):
        """Test creating a semantic frame."""
        frame = SemanticFrame(
            predicate="wants",
            arguments={"agent": "user", "theme": "answer"},
        )
        assert frame.predicate == "wants"


# ==================== Clarification System Tests ====================

class TestClarificationManager:
    """Test suite for ClarificationManager."""

    @pytest.fixture
    def manager(self):
        """Create ClarificationManager instance."""
        return ClarificationManager()

    @pytest.mark.unit
    def test_manager_creation(self, manager):
        """Test creating clarification manager."""
        assert manager is not None


class TestAmbiguityDetector:
    """Test suite for AmbiguityDetector."""

    @pytest.fixture
    def detector(self):
        """Create AmbiguityDetector instance."""
        return AmbiguityDetector()

    @pytest.mark.unit
    def test_detector_creation(self, detector):
        """Test creating ambiguity detector."""
        assert detector is not None


class TestQuestionGenerator:
    """Test suite for QuestionGenerator."""

    @pytest.fixture
    def generator(self):
        """Create QuestionGenerator instance."""
        return QuestionGenerator()

    @pytest.mark.unit
    def test_generator_creation(self, generator):
        """Test creating question generator."""
        assert generator is not None
