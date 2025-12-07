"""
Unit tests for Evidence System

Tests evidence tracking, validation, source trust, and citation requirements.
Updated to match current API (Phase 2 Enhancement).
"""

import pytest
from agents.core.evidence_system import (
    SourceType,
    TrustLevel,
    SourceProfile,
    EvidenceItem,
    CitationRequirement,
    EvidenceValidation,
    SourceTrustRegistry,
    ConfidenceChain,
    HallucinationGuard,
    EvidenceValidator,
    ConflictResolver,
)


class TestSourceProfile:
    """Test suite for SourceProfile dataclass."""

    @pytest.mark.unit
    def test_source_profile_creation(self):
        """Test creating a source profile."""
        profile = SourceProfile(
            source_id="src_001",
            source_type=SourceType.FACT,
            trust_level=TrustLevel.HIGH,
        )

        assert profile.source_id == "src_001"
        assert profile.source_type == SourceType.FACT
        assert profile.trust_level == TrustLevel.HIGH

    @pytest.mark.unit
    def test_trust_score_calculation(self):
        """Test trust score from trust level."""
        profile = SourceProfile(
            source_id="src_002",
            source_type=SourceType.TOOL_RESULT,
            trust_level=TrustLevel.MEDIUM,
        )

        assert profile.trust_score == pytest.approx(0.75, rel=0.01)

    @pytest.mark.unit
    def test_trust_score_with_track_record(self):
        """Test trust score adjusted by success/failure counts."""
        profile = SourceProfile(
            source_id="src_003",
            source_type=SourceType.EXTERNAL_API,
            trust_level=TrustLevel.MEDIUM,
            success_count=9,
            failure_count=1,
        )

        # Base is 0.75, adjusted by 70% base + 30% success rate (0.9)
        expected = 0.7 * 0.75 + 0.3 * 0.9
        assert profile.trust_score == pytest.approx(expected, rel=0.01)


class TestEvidenceItem:
    """Test suite for EvidenceItem dataclass."""

    @pytest.fixture
    def source_profile(self):
        """Create a source profile for testing."""
        return SourceProfile(
            source_id="test_source",
            source_type=SourceType.FACT,
            trust_level=TrustLevel.HIGH,
        )

    @pytest.mark.unit
    def test_evidence_item_creation(self, source_profile):
        """Test creating an evidence item."""
        evidence = EvidenceItem(
            evidence_id="ev_001",
            content="Temperature has risen 1C",
            source=source_profile,
            confidence=0.95,
        )

        assert evidence.evidence_id == "ev_001"
        assert evidence.confidence == 0.95
        assert evidence.source.trust_level == TrustLevel.HIGH

    @pytest.mark.unit
    def test_effective_confidence(self, source_profile):
        """Test effective confidence calculation."""
        evidence = EvidenceItem(
            evidence_id="ev_002",
            content="Observation data",
            source=source_profile,
            confidence=1.0,
            reasoning_depth=0,
        )

        # No depth decay, trust score is HIGH (0.95)
        assert evidence.effective_confidence == pytest.approx(0.95, rel=0.01)

    @pytest.mark.unit
    def test_effective_confidence_with_depth(self, source_profile):
        """Test effective confidence decays with reasoning depth."""
        evidence = EvidenceItem(
            evidence_id="ev_003",
            content="Derived conclusion",
            source=source_profile,
            confidence=1.0,
            reasoning_depth=2,
        )

        # 5% decay per step: 0.95^2 = 0.9025, times trust 0.95
        expected = 1.0 * 0.95 * (0.95 ** 2)
        assert evidence.effective_confidence == pytest.approx(expected, rel=0.01)


class TestCitationRequirement:
    """Test suite for CitationRequirement dataclass."""

    @pytest.mark.unit
    def test_default_requirements(self):
        """Test default citation requirements."""
        req = CitationRequirement()

        assert req.min_citations == 1
        assert req.min_confidence == 0.5
        assert req.min_source_trust == 0.5
        assert req.allow_llm_only is False

    @pytest.mark.unit
    def test_strict_requirements(self):
        """Test strict citation requirements."""
        req = CitationRequirement(
            min_citations=3,
            min_confidence=0.9,
            min_source_trust=0.8,
            require_multiple_sources=True,
            allow_llm_only=False,
        )

        assert req.min_citations == 3
        assert req.require_multiple_sources is True


class TestSourceTrustRegistry:
    """Test suite for SourceTrustRegistry."""

    @pytest.fixture
    def registry(self):
        """Create SourceTrustRegistry instance."""
        return SourceTrustRegistry()

    @pytest.mark.unit
    def test_registry_initialization(self, registry):
        """Test registry has default sources."""
        assert registry.sources is not None

    @pytest.mark.unit
    def test_register_source(self, registry):
        """Test registering a new source."""
        profile = SourceProfile(
            source_id="custom_api",
            source_type=SourceType.EXTERNAL_API,
            trust_level=TrustLevel.MEDIUM,
        )

        registry.sources["custom_api"] = profile
        assert "custom_api" in registry.sources


class TestEvidenceValidator:
    """Test suite for EvidenceValidator."""

    @pytest.fixture
    def validator(self):
        """Create EvidenceValidator instance."""
        return EvidenceValidator()

    @pytest.mark.unit
    def test_validator_creation(self, validator):
        """Test creating validator."""
        assert validator is not None


class TestConflictResolver:
    """Test suite for ConflictResolver."""

    @pytest.fixture
    def resolver(self):
        """Create ConflictResolver instance with a trust registry."""
        registry = SourceTrustRegistry()
        return ConflictResolver(trust_registry=registry)

    @pytest.mark.unit
    def test_resolver_creation(self, resolver):
        """Test creating resolver."""
        assert resolver is not None


class TestHallucinationGuard:
    """Test suite for HallucinationGuard."""

    @pytest.fixture
    def guard(self):
        """Create HallucinationGuard instance."""
        return HallucinationGuard()

    @pytest.mark.unit
    def test_guard_creation(self, guard):
        """Test creating hallucination guard."""
        assert guard is not None


class TestConfidenceChain:
    """Test suite for ConfidenceChain."""

    @pytest.fixture
    def chain(self):
        """Create ConfidenceChain instance."""
        return ConfidenceChain()

    @pytest.mark.unit
    def test_chain_creation(self, chain):
        """Test creating confidence chain."""
        assert chain is not None
