"""Architecture selection policy with decision matrix defaults.

This module codifies when to prefer 4×, 8×, 16×, or 32× variants based on
latency tolerance, accuracy priority, risk level, and cost sensitivity.
The thresholds mirror the guidance in the scalability analysis where 8× is
the default for balanced cost/benefit, 4× is for latency-constrained
workloads, and 16×/32× are reserved for high-accuracy or high-risk contexts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .decision_model import RiskLevel


class ArchitectureVariant(Enum):
    """Supported architecture sizes."""

    FOUR_X = 4
    EIGHT_X = 8
    SIXTEEN_X = 16
    THIRTY_TWO_X = 32


@dataclass
class ArchitectureDecisionInput:
    """Inputs used to decide which architecture variant to run."""

    latency_tolerance_ms: int
    accuracy_priority: float  # 0.0 (latency-first) to 1.0 (accuracy-first)
    risk_level: RiskLevel
    cost_sensitivity: float  # 0.0 (cost-insensitive) to 1.0 (cost-constrained)
    availability_constraints: Optional[bool] = False


@dataclass
class ArchitectureRecommendation:
    """Decision output describing the chosen variant and rationale."""

    variant: ArchitectureVariant
    reason: str
    estimated_latency_ms: int
    logic_weight: float
    accuracy_gain: float


class ArchitectureSelector:
    """Applies the policy defaults from the scalability analysis."""

    def __init__(
        self, default_variant: ArchitectureVariant = ArchitectureVariant.EIGHT_X
    ):
        self.default_variant = default_variant

    def recommend(
        self, decision: ArchitectureDecisionInput
    ) -> ArchitectureRecommendation:
        """Return the architecture variant that best matches the policy inputs."""

        # Latency and cost sensitive flows downshift to 4× unless risk/accuracy demand more
        if decision.latency_tolerance_ms <= 300 or decision.cost_sensitivity >= 0.75:
            return ArchitectureRecommendation(
                variant=ArchitectureVariant.FOUR_X,
                reason="Latency/cost sensitive workload benefits from 4× baseline.",
                estimated_latency_ms=250,
                logic_weight=0.60,
                accuracy_gain=1.0,
            )

        # Ultra-accuracy cases with relaxed latency and low cost sensitivity reach 32×
        if (
            decision.accuracy_priority >= 0.9
            and decision.latency_tolerance_ms >= 1500
            and decision.cost_sensitivity <= 0.4
            and not decision.availability_constraints
        ):
            return ArchitectureRecommendation(
                variant=ArchitectureVariant.THIRTY_TWO_X,
                reason="Extreme accuracy with relaxed latency/cost enables 32× variant.",
                estimated_latency_ms=1600,
                logic_weight=0.88,
                accuracy_gain=1.22,
            )

        # Accuracy-first or high-risk scenarios escalate to 16× if latency budget allows
        if decision.accuracy_priority >= 0.8 or decision.risk_level in {
            RiskLevel.HIGH,
            RiskLevel.CRITICAL,
        }:
            if decision.latency_tolerance_ms >= 850:
                return ArchitectureRecommendation(
                    variant=ArchitectureVariant.SIXTEEN_X,
                    reason="High accuracy/risk with adequate latency budget favors 16× stack.",
                    estimated_latency_ms=850,
                    logic_weight=0.82,
                    accuracy_gain=1.18,
                )
            # If timing is tighter, prefer 8× but annotate why 16× was avoided
            return ArchitectureRecommendation(
                variant=ArchitectureVariant.EIGHT_X,
                reason="High accuracy/risk detected but latency budget keeps us at 8×.",
                estimated_latency_ms=450,
                logic_weight=0.75,
                accuracy_gain=1.12,
            )

        # Balanced default
        return ArchitectureRecommendation(
            variant=self.default_variant,
            reason="Balanced workload defaults to 8× per scalability guidance.",
            estimated_latency_ms=450,
            logic_weight=0.75,
            accuracy_gain=1.12,
        )
