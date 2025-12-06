"""
Categorical Engine - Re-export module for backward compatibility.

This module re-exports all categorical engine components from the canonical
implementation in agents.core_logic.categorical_engine.

The comprehensive categorical logic implementation with syllogistic reasoning
is located in agents/core_logic/categorical_engine.py.
"""

# Re-export from canonical implementation
from agents.core_logic.categorical_engine import (
    CategoricalEngine,
    CategoricalStatement,
    Figure,
    StatementType,
    Syllogism,
    SyllogismValidation,
    parse_categorical_statement,
    parse_syllogism,
    validate_barbara,
)


# Backward compatibility aliases
class SyllogismType:
    """
    Backward compatibility enum for syllogism forms.
    
    New code should use Figure and mood strings from agents.core_logic.categorical_engine.
    """
    BARBARA = "AAA-1"
    CELARENT = "EAE-1"
    DARII = "AII-1"
    FERIO = "EIO-1"
    CESARE = "EAE-2"
    CAMESTRES = "AEE-2"
    FESTINO = "EIO-2"
    BAROCO = "AOO-2"


class SyllogismResult:
    """
    Backward compatibility wrapper for SyllogismValidation.
    
    New code should use SyllogismValidation from agents.core_logic.categorical_engine.
    """
    
    def __init__(
        self,
        valid: bool,
        form: str | None = None,
        explanation: str = "",
        confidence: float = 1.0,
    ):
        self.valid = valid
        self.form = form
        self.explanation = explanation
        self.confidence = confidence
    
    @classmethod
    def from_validation(cls, result: SyllogismValidation) -> "SyllogismResult":
        """Convert a SyllogismValidation to SyllogismResult for backward compatibility."""
        form_str = f"{result.mood}-{result.figure.value}" if result.mood else None
        return cls(
            valid=result.is_valid,
            form=form_str,
            explanation=result.explanation,
            confidence=result.confidence,
        )


__all__ = [
    # Canonical exports
    "CategoricalEngine",
    "CategoricalStatement",
    "Figure",
    "StatementType",
    "Syllogism",
    "SyllogismValidation",
    "parse_categorical_statement",
    "parse_syllogism",
    "validate_barbara",
    # Backward compatibility
    "SyllogismType",
    "SyllogismResult",
]
