from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


class ProcessStage(Enum):
    QUESTION_ANALYSIS = auto()
    CONCEPT_EXTRACTION = auto()
    CONSTRAINT_CHECK = auto()
    PLAN_VALIDATION = auto()
    ACTION_MAPPING = auto()
    SAFETY_REVIEW = auto()
    OUTPUT_GENERATION = auto()


@dataclass
class StageResult:
    stage: ProcessStage
    passed: bool
    confidence: float
    evidence: List[str] = field(default_factory=list)
    needs_clarification: Optional[str] = None


class ProcessGate:
    """Validates sequential process stages before allowing output."""

    def __init__(self, min_confidence: float = 0.6) -> None:
        self.min_confidence = min_confidence
        self._results: Dict[ProcessStage, StageResult] = {}
        self._failed_stage: Optional[ProcessStage] = None

    def record_stage(self, result: StageResult) -> None:
        if self._failed_stage is not None:
            raise ValueError(f"{self._failed_stage.name} failed; cannot continue")

        expected_stage = list(ProcessStage)[len(self._results)]
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
        return {
            "stages_completed": len(self._results),
            "stages": [stage.name for stage in self._results.keys()],
            "confidence_scores": {
                stage.name: res.confidence for stage, res in self._results.items()
            },
        }
