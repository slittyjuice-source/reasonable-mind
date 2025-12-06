"""
Logic Engine - Re-export module for backward compatibility.

This module re-exports all logic engine components from the canonical
implementation in agents.core_logic.logic_engine.

The comprehensive logic engine implementation with truth table evaluation
and pattern matching is located in agents/core_logic/logic_engine.py.
"""

from typing import List, Optional

# Re-export from canonical implementation
from agents.core_logic.logic_engine import (
    LogicEngine as CoreLogicEngine,
    LogicForm,
    TruthValue,
    Proposition,
    LogicalArgument,
    ValidationResult,
    parse_argument,
    validate_modus_ponens,
    validate_modus_tollens,
)

# Backward compatibility aliases
ArgumentForm = LogicForm  # Legacy name


class LogicResult:
    """
    Backward compatibility wrapper for ValidationResult.
    
    New code should use ValidationResult from agents.core_logic.logic_engine.
    """
    
    def __init__(
        self,
        valid: bool,
        form: LogicForm | None = None,
        explanation: str = "",
        confidence: float = 1.0,
    ):
        self.valid = valid
        self.form = form
        self.explanation = explanation
        self.confidence = confidence
    
    @classmethod
    def from_validation_result(cls, result: ValidationResult) -> "LogicResult":
        """Convert a ValidationResult to LogicResult for backward compatibility."""
        return cls(
            valid=result.is_valid,
            form=result.form,
            explanation=result.explanation,
            confidence=result.confidence,
        )


class LogicEngine(CoreLogicEngine):
    """
    Backward-compatible LogicEngine with legacy validate_argument() method.
    
    Extends CoreLogicEngine to provide the old API while using the new implementation.
    """
    
    def validate_argument(self, premises: List[str], conclusion: str) -> LogicResult:
        """
        Validate an argument using formal logic rules.
        
        Legacy method for backward compatibility. New code should use validate().
        
        Args:
            premises: List of premise statements
            conclusion: Conclusion statement
            
        Returns:
            LogicResult with validity, form, and explanation
        """
        # Convert to new API format
        argument = LogicalArgument(
            premises=[Proposition(symbol=f"P{i}", statement=p) for i, p in enumerate(premises)],
            conclusion=Proposition(symbol="C", statement=conclusion),
        )
        
        # Use the core validate method
        result = self.validate(argument)
        
        # Convert back to legacy format
        return LogicResult.from_validation_result(result)


__all__ = [
    # Canonical exports
    "LogicEngine",
    "LogicForm",
    "TruthValue",
    "Proposition",
    "LogicalArgument",
    "ValidationResult",
    "parse_argument",
    "validate_modus_ponens",
    "validate_modus_tollens",
    # Backward compatibility
    "ArgumentForm",
    "LogicResult",
]
