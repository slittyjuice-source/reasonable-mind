# Changelog - agents/tools

All notable changes to the tools module will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **extended_thinking.py**: ExtendedThinkingTool for multi-layer chain-of-thought reasoning
  - 6-step thinking process: Question Analysis → Key Concepts → Multi-Layer Analysis → Strategy Selection → Option Evaluation → Consensus Synthesis
  - Configurable layers (4, 8, 16, or 32) with specialized focuses per layer
  - Logic-weighted consensus (default 75% weight to logic layers)
  - Depth parameter (1-5) for controlling analysis thoroughness
  - Thinking history tracking for pattern learning

- **extended_thinking.py**: Query-aware confidence scoring (2025-12-04)
  - `_analyze_query_confidence()` modulates confidence based on query characteristics
  - Boosts confidence for: clear logical structure, domain terminology, available context
  - Reduces confidence for: vague language, very short queries, complex multi-part questions
  - Result: Confidence varies meaningfully (69%-79% range vs. uniform 72%)

- **extended_thinking.py**: Enhanced option evaluation
  - Scores options by semantic overlap with query terms
  - Detects absolute language (always/never) and applies penalty
  - Rewards hedging language (likely/probably) for accuracy
  - Considers option length and specificity

- **extended_thinking.py**: Depth bonus system
  - Each additional layer beyond first adds +2.5% confidence (max 10% total)
  - Logic layer disagreement penalty: -10% if agreement < 70%
  - Result: Depth 1→3 shows +7% gain, Depth 3→5 shows +2.3% additional gain

- **extended_thinking.py**: Modular reasoning system
  - `enable_module()` for activating specialized reasoning modules
  - Available modules: watson_glaser (critical thinking with cognitive templates)
  - Module state management with per-module initialization and handlers

- **extended_thinking.py**: WatsonGlaserThinkingTool specialization
  - Curriculum learning with complexity gating (levels 1-4)
  - unlock_complexity() based on accuracy thresholds (70%, 80%, 90%)
  - Cognitive template matching for assumptions, inferences, deductions, interpretations, evaluations
  - Accuracy tracking and adaptive recommendations

- **extended_thinking_integration.ipynb**: Interactive demonstration notebook
  - 10 sections covering all ExtendedThinkingTool features
  - Examples: basic reasoning, layer analysis, Watson Glaser, complex decisions
  - History tracking, meta-analysis, and tool schema generation
  - Test results documenting confidence variation and depth effectiveness

### Changed
- **extended_thinking.py**: Improved `_synthesize_consensus()` to prioritize logic layers
  - Logic layers weighted at 75% vs. 25% for other layers
  - Depth bonus added to base confidence
  - Agreement checking among logic layers with disagreement penalty

- **extended_thinking.py**: Enhanced `_multi_layer_analysis()` with query modulation
  - Base confidence increases with layer depth (0.6 + layer*0.05)
  - Modulated by query confidence factor (0.7-1.3 range)
  - Final confidence clipped to valid range (0.3-0.95)

### Fixed
- **extended_thinking_integration.ipynb**: Fixed AttributeError accessing `et_tool.history`
  - Corrected to `et_tool.thinking_history` (actual attribute name)
  - Added defensive checks in debug cell to validate variable existence
  - Added execution order warning for sequential cell execution

### Known Issues
- **extended_thinking.py**: Bare `except` clauses (lines 328, 493)
  - Silently catches all exceptions in handler callbacks
  - TODO: Catch specific exceptions and log failures

## [1.0.0] - 2024-XX-XX

### Added
- Initial release of tools module
- ExtendedThinkingTool with basic chain-of-thought reasoning
- Support for 4-layer and 8-layer configurations
