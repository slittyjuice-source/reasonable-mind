"""
Process validity gate enforcing 7-stage reasoning pipeline.

Stages: question analysis → concepts → layer views → strategies → 
        module insights → option evaluation → consensus

A reasonable mind only outputs when all stages pass with sufficient confidence.
"""

from enum import Enum
from typing import Optional, Dict, List
from dataclasses import dataclass, field


class ProcessStage(Enum):
    """Mandatory reasoning stages in order."""
    QUESTION_ANALYSIS = 1
    CONCEPT_EXTRACTION = 2
    LAYER_VIEWS = 3
    STRATEGY_FORMULATION = 4
    MODULE_INSIGHTS = 5
    OPTION_EVALUATION = 6
    CONSENSUS = 7


@dataclass
class StageResult:
    """Result of completing a reasoning stage."""
    stage: ProcessStage
    passed: bool
    confidence: float  # 0.0-1.0
    evidence: List[str] = field(default_factory=list)
    needs_clarification: Optional[str] = None
    
    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")


class ProcessGate:
    """
    Enforces 7-step reasoning pipeline with confidence thresholds.
    
    Usage:
        gate = ProcessGate(min_confidence=0.7)
        gate.record_stage(StageResult(ProcessStage.QUESTION_ANALYSIS, True, 0.9, ["parsed intent"]))
        ...
        can_proceed, reason = gate.can_output()
    """
    
    def __init__(self, min_confidence: float = 0.7):
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be 0.0-1.0, got {min_confidence}")
        self.min_confidence = min_confidence
        self._completed: Dict[ProcessStage, StageResult] = {}
    
    def record_stage(self, result: StageResult) -> None:
        """
        Record completion of a stage.
        
        Raises ValueError if previous stage not completed (enforces order).
        """
        if result.stage.value > 1:
            prev = ProcessStage(result.stage.value - 1)
            if prev not in self._completed:
                raise ValueError(f"Cannot proceed to {result.stage.name}: {prev.name} not completed")
            if not self._completed[prev].passed:
                raise ValueError(f"Cannot proceed to {result.stage.name}: {prev.name} failed")
        self._completed[result.stage] = result
    
    def can_output(self) -> tuple[bool, Optional[str]]:
        """
        Check if all stages passed with sufficient confidence.
        
        Returns (can_proceed, reason_if_blocked).
        """
        # Check all stages completed
        for stage in ProcessStage:
            if stage not in self._completed:
                return False, f"Stage {stage.name} not completed"
        
        # Check consensus reached
        if not self._completed[ProcessStage.CONSENSUS].passed:
            return False, "Consensus stage did not pass"
        
        # Check average confidence
        avg_confidence = sum(r.confidence for r in self._completed.values()) / len(self._completed)
        if avg_confidence < self.min_confidence:
            return False, f"Average confidence {avg_confidence:.2f} below threshold {self.min_confidence}"
        
        # Check for unresolved clarifications
        for result in self._completed.values():
            if result.needs_clarification:
                return False, f"Stage {result.stage.name} needs clarification: {result.needs_clarification}"
        
        return True, None
    
    def get_blocking_stage(self) -> Optional[ProcessStage]:
        """Get the first incomplete or failed stage."""
        for stage in ProcessStage:
            if stage not in self._completed:
                return stage
            if not self._completed[stage].passed:
                return stage
        return None
    
    def reset(self) -> None:
        """Reset for a new reasoning cycle."""
        self._completed.clear()
    
    def to_audit_record(self) -> Dict:
        """Generate audit-friendly representation."""
        return {
            "min_confidence": self.min_confidence,
            "stages_completed": len(self._completed),
            "stages": {
                stage.name: {
                    "passed": result.passed,
                    "confidence": result.confidence,
                    "needs_clarification": result.needs_clarification
                }
                for stage, result in self._completed.items()
            }
        }
