"""Process validity gate used by tests in agents/tests/test_governance.py.

The gate tracks a fixed sequence of stages and enforces:
- Stages must be recorded in order (no skipping)
- A failed stage blocks subsequent stages
- Output is allowed only when all stages pass with adequate confidence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


class ProcessStage(Enum):
    """Ordered stages for governance validation.

    Keep this enum stable — tests rely on its iteration order.
    """

    QUESTION_ANALYSIS = auto()
    CONCEPT_EXTRACTION = auto()
    CONSTRAINT_CHECK = auto()
    PLAN_VALIDATION = auto()
    ACTION_MAPPING = auto()
    SAFETY_REVIEW = auto()
    OUTPUT_GENERATION = auto()


@dataclass
class StageResult:
    """Result for a single process stage."""

    stage: ProcessStage
    passed: bool
    confidence: float
    evidence: List[str] = field(default_factory=list)
    needs_clarification: Optional[str] = None


class ProcessGate:
    """Enforces sequential stage completion and confidence thresholds.

    Example usage:
        gate = ProcessGate(min_confidence=0.6)
        gate.record_stage(StageResult(ProcessStage.QUESTION_ANALYSIS, True, 0.9))
        ...
        ok, reason = gate.can_output()
    """

    def __init__(self, min_confidence: float = 0.6) -> None:
        self.min_confidence = float(min_confidence)
        self._results: Dict[ProcessStage, StageResult] = {}
        self._failed_stage: Optional[ProcessStage] = None

    def record_stage(self, result: StageResult) -> None:
        """Record a stage result, enforcing ordering and halting on failure.

        Raises ValueError if stages are out-of-order, already recorded, or if a
        previous stage failed.
        """

        if self._failed_stage is not None:
            raise ValueError(f"{self._failed_stage.name} failed; cannot continue")

        stages = list(ProcessStage)
        if len(self._results) >= len(stages):
            raise ValueError("All stages already recorded")

        expected_stage = stages[len(self._results)]
        if result.stage != expected_stage:
            raise ValueError(
                f"{expected_stage.name} not completed before {result.stage.name}"
            )

        if result.stage in self._results:
            raise ValueError(f"{result.stage.name} already recorded")

        # store the result and mark failure if it did not pass
        self._results[result.stage] = result
        if not result.passed:
            self._failed_stage = result.stage

    def can_output(self) -> Tuple[bool, Optional[str]]:
        """Return (allowed, reason)."""

        if self._failed_stage is not None:
            return False, f"{self._failed_stage.name} failed"

        stages = list(ProcessStage)
        if len(self._results) != len(stages):
            missing = [s.name for s in stages if s not in self._results]
            return False, f"Missing stages: {', '.join(missing)}"

        for stage in stages:
            res = self._results[stage]
            if res.needs_clarification:
                return False, f"{stage.name} needs clarification"
            if res.confidence < self.min_confidence:
                return False, f"{stage.name} confidence below threshold"
            if not res.passed:
                return False, f"{stage.name} failed"

        return True, None

    def to_audit_record(self) -> Dict[str, object]:
        """Return an audit-friendly representation of the gate state."""

        return {
            "stages_completed": len(self._results),
            "stages": [stage.name for stage in self._results.keys()],
            "confidence_scores": {
                stage.name: res.confidence for stage, res in self._results.items()
            },
            "min_confidence": self.min_confidence,
        }
