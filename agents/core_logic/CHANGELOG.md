# Changelog - agents/core_logic

All notable changes to the core logic module will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **logic_engine.py**: Deterministic propositional logic evaluation engine
  - Truth table validation for arguments with ≤5 variables (2^5 = 32 rows)
  - Pattern matching against known valid/invalid forms
  - Heuristic validation as fallback for complex arguments
  - Support for logical operators: →, ∧, ∨, ¬, ↔
  - Counterexample generation for invalid arguments

- **logic_engine.py**: LogicForm enum with valid and invalid argument forms
  - Valid: Modus Ponens, Modus Tollens, Hypothetical Syllogism, Disjunctive Syllogism, etc.
  - Invalid: Affirming Consequent, Denying Antecedent, Undistributed Middle, etc.

- **logic_engine.py**: ValidationResult dataclass with rich feedback
  - Includes is_valid, form_identified, truth_table_valid, counterexample
  - Confidence scoring (1.0 for deterministic, <1.0 for heuristic)
  - Method tracking (pattern_match, truth_table, heuristic)

- **logic_engine.py**: Convenience functions
  - validate_modus_ponens()
  - validate_modus_tollens()
  - parse_argument() for converting premises/conclusion to LogicalArgument

### Known Issues
- **logic_engine.py**: Use of `eval()` for expression evaluation (line 415)
  - Security risk: Code injection if expressions aren't properly sanitized
  - Impact: Mitigated by tokenization, but still uses Python eval()
  - Recommendation: Replace with AST-based evaluator or safe expression parser
  - Workaround: Only use with trusted input or in sandboxed environment

- **logic_engine.py**: Bare `except` clause catches all exceptions (line 416-418)
  - Returns False on any error, hiding potential bugs
  - TODO: Catch specific exceptions (SyntaxError, NameError, ValueError)

- **logic_engine.py**: Simplified implication conversion (line 420-443)
  - `_convert_implications()` assumes no nested implications
  - May fail on complex nested expressions like `((P → Q) → R) → S`
  - TODO: Implement proper recursive parser for nested logical operators

### Changed
- None

## [1.0.0] - 2024-XX-XX

### Added
- Initial release of core_logic module
- Pattern matching for common argument forms
- Truth table validation for simple propositional arguments
- Heuristic fallback for complex cases
