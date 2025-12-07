"""
Process validity gate used by tests in agents/tests/test_governance.py.

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
    """Ordered stages for governance validation."""

    QUESTION_ANALYSIS = auto()
    CONCEPT_EXTRACTION = auto()
    EVIDENCE_GATHERING = auto()
    REASONING = auto()
    PLAN_CONSTRUCTION = auto()
    EXECUTION = auto()
    REVIEW = auto()
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
    """Enforces sequential stage completion and confidence thresholds."""

    def __init__(self, min_confidence: float = 0.7) -> None:
        self.min_confidence = min_confidence
        self._results: Dict[ProcessStage, StageResult] = {}
        self._ordered_stages = list(ProcessStage)

    def record_stage(self, result: StageResult) -> None:
        """Record a stage result, enforcing ordering and prior success."""

        if result.stage in self._results:
            raise ValueError(f"{result.stage.name} already recorded")

        expected_stage = self._ordered_stages[len(self._results)]
        if result.stage != expected_stage:
            raise ValueError(f"{expected_stage.name} not completed")

        # Block progression if previous stage failed
        if len(self._results) > 0:
            prev_stage = self._ordered_stages[len(self._results) - 1]
            prev_result = self._results[prev_stage]
            if not prev_result.passed:
                raise ValueError(f"{prev_stage.name} failed")

        self._results[result.stage] = result

    def can_output(self) -> Tuple[bool, Optional[str]]:
        """Determine if output is allowed based on recorded stages."""

        # All stages must be present
        if len(self._results) != len(self._ordered_stages):
            missing = [s.name for s in self._ordered_stages if s not in self._results]
            return False, f"Missing stages: {', '.join(missing)}"

        for stage in self._ordered_stages:
            res = self._results[stage]
            if not res.passed:
                return False, f"{stage.name} failed"
            if res.confidence < self.min_confidence:
                return False, "below threshold"
            if res.needs_clarification:
                return False, "needs clarification"
    """Validates sequential process stages before allowing output."""

    def __init__(self, min_confidence: float = 0.6) -> None:
        self.min_confidence = min_confidence
        self._results: Dict[ProcessStage, StageResult] = {}
        self._failed_stage: Optional[ProcessStage] = None

    def record_stage(self, result: StageResult) -> None:
        if self._failed_stage is not None:
            raise ValueError(f"{self._failed_stage.name} failed; cannot continue")

        stages = list(ProcessStage)
        if len(self._results) >= len(stages):
            raise ValueError("All stages already recorded")

        expected_stage = stages[len(self._results)]
        if result.stage != expected_stage:
            raise ValueError(f"{expected_stage.name} not completed before {result.stage.name}")

        self._results[result.stage] = result
        if not result.passed:
            self._failed_stage = result.stage

    def can_output(self) -> Tuple[bool, Optional[str]]:
        if self._failed_stage is not None:
            return False, f"{self._failed_stage.name} failed"

        if len(self._results) != len(ProcessStage):
            missing = [s.name for s in ProcessStage if s not in self._results]
            return False, f"Missing stages: {', '.join(missing)}"

        for stage, result in self._results.items():
            if result.needs_clarification:
                return False, f"{stage.name} needs clarification"
            if result.confidence < self.min_confidence:
                return False, f"{stage.name} confidence below threshold"

        return True, None

    def to_audit_record(self) -> Dict[str, object]:
        """Return audit-friendly representation."""

        return {
            "stages_completed": len(self._results),
            "stages": [stage.name for stage in self._results],
            "min_confidence": self.min_confidence,
        return {
            "stages_completed": len(self._results),
            "stages": [stage.name for stage in self._results.keys()],
            "confidence_scores": {
                stage.name: res.confidence for stage, res in self._results.items()
            },
        }
