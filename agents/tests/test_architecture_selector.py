from agents.core.architecture_selector import (
    ArchitectureDecisionInput,
    ArchitectureSelector,
    ArchitectureVariant,
)
from agents.core.decision_model import RiskLevel


def test_latency_sensitive_prefers_4x():
    selector = ArchitectureSelector()
    decision = ArchitectureDecisionInput(
        latency_tolerance_ms=250,
        accuracy_priority=0.3,
        risk_level=RiskLevel.LOW,
        cost_sensitivity=0.9,
    )

    rec = selector.recommend(decision)

    assert rec.variant is ArchitectureVariant.FOUR_X
    assert "Latency" in rec.reason


def test_high_accuracy_and_risk_reaches_16x_when_budget_allows():
    selector = ArchitectureSelector()
    decision = ArchitectureDecisionInput(
        latency_tolerance_ms=1000,
        accuracy_priority=0.9,
        risk_level=RiskLevel.HIGH,
        cost_sensitivity=0.2,
    )

    rec = selector.recommend(decision)

    assert rec.variant is ArchitectureVariant.SIXTEEN_X
    assert rec.logic_weight > 0.8


def test_default_balances_to_8x():
    selector = ArchitectureSelector()
    decision = ArchitectureDecisionInput(
        latency_tolerance_ms=600,
        accuracy_priority=0.6,
        risk_level=RiskLevel.MEDIUM,
        cost_sensitivity=0.4,
    )

    rec = selector.recommend(decision)

    assert rec.variant is ArchitectureVariant.EIGHT_X
    assert "Balanced" in rec.reason


def test_extreme_accuracy_with_low_cost_sensitivity_enables_32x():
    selector = ArchitectureSelector()
    decision = ArchitectureDecisionInput(
        latency_tolerance_ms=2000,
        accuracy_priority=0.95,
        risk_level=RiskLevel.MEDIUM,
        cost_sensitivity=0.2,
    )

    rec = selector.recommend(decision)

    assert rec.variant is ArchitectureVariant.THIRTY_TWO_X
    assert rec.logic_weight > 0.85

