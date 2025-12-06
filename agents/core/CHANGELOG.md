# Changelog - agents/core

All notable changes to the core reasoning module will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **logic_orchestrator.py**: New LogicOrchestrator class as single entry point for deterministic reasoning
  - Coordinates between CategoricalEngine, InferenceEngine, and FallacyDetector
  - Routes arguments based on ArgumentType (categorical, propositional, mixed, unknown)
  - Returns unified LogicAnalysisResult with validity, fallacies, and confidence scores
  - Supports analyze(), check_validity(), and detect_fallacies_only() methods
- **logic_orchestrator.py**: StructuredArgument dataclass for explicit premise/conclusion representation
- **logic_orchestrator.py**: ArgumentType enum for routing strategy
- **logic_orchestrator.py**: Factory functions create_orchestrator() and analyze_argument()

- **categorical_engine.py**: Expanded syllogism support to 8 forms
  - Added second-figure forms: Cesare (EAE-2), Camestres (AEE-2), Festino (EIO-2), Baroco (AOO-2)
  - Each form includes pattern, description, and example in valid_forms dictionary

- **curriculum_system.py**: Complete curriculum and evaluation system
  - DifficultyLevel enum (EASY → MEDIUM → HARD → EXPERT)
  - LogicDataset and ArgumentDataset with tiered examples
  - EvalHarness for running evaluations with multiple scoring methods
  - CurriculumLearner with adaptive progression based on performance
  - Supports run_curriculum(), compare_runs(), and performance tracking

- **observability_system.py**: Comprehensive observability infrastructure
  - Distributed tracing with Tracer, Trace, and Span classes
  - Metrics registry with Counter, Histogram, and Gauge types
  - EventLogger with structured TraceEvent logging
  - TokenCounter for tracking LLM usage
  - ObservabilitySystem facade with pre-configured common metrics

- **role_system.py**: Role adaptation system for persona-based reasoning
  - RolePersona dataclass with expertise levels and communication styles
  - Built-in personas: lawyer, scientist, tutor, socratic, critic
  - RoleAdapter for applying vocabulary substitutions and constraint checks
  - RoleBasedReasoner for persona-adapted reasoning flows
  - PersonaManager for creating custom personas

### Changed
- **categorical_engine.py**: validate_syllogism() now detects form codes but only validates 4 forms
  - Forms 5-8 (Cesare, Camestres, Festino, Baroco) are defined but not yet validated
  - Returns proper SyllogismResult with form identification

### Known Issues
- **categorical_engine.py**: `_is_first_figure()` always returns True (line 190)
  - Impact: All syllogisms incorrectly validated as first-figure
  - TODO: Implement proper figure detection or raise NotImplementedError

- **logic_orchestrator.py**: `_analyze_categorical()` returns CATEGORICAL_PARSING_NOT_IMPLEMENTED
  - Natural language premise parsing not yet implemented
  - TODO: Parse premises to CategoricalProposition objects

- **logic_orchestrator.py**: `analyze_text()` returns PARSE_NOT_IMPLEMENTED
  - Argument extraction from natural language not implemented
  - TODO: Integrate with semantic_parser.py or LLM-based extraction

- **logic_orchestrator.py**: `_analyze_propositional()` mutates InferenceEngine state
  - Violates INVARIANT #2 (side-effect free)
  - Facts accumulate across multiple analyze() calls
  - TODO: Create new engine instance per analysis or add reset() method

## [1.0.0] - 2024-XX-XX

### Added
- Initial release of core reasoning modules
- CategoricalEngine for syllogistic logic validation
- Basic support for Barbara, Celarent, Darii, Ferio forms
