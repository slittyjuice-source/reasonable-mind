# Reasonable Mind

A neuro-symbolic reasoning framework combining rigorous logical foundations with extended thinking capabilities.

## Overview

Reasonable Mind is a modular agent architecture that implements:

- **Logic Engine**: Categorical and propositional reasoning
- **Extended Thinking**: Chain-of-thought with structured deliberation
- **Uncertainty Quantification**: Calibrated confidence estimation
- **Evidence Synthesis**: Multi-source reasoning with source trust
- **Safety Systems**: Hallucination mitigation and robustness checks

## Quick Start

```bash
# Clone and setup
cd ~/Documents/GitHub/reasonable-mind
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest agents/tests/ -v
```

## Project Structure

```
reasonable-mind/
├── agents/               # Core agent framework
│   ├── core/            # Reasoning engines and systems
│   ├── core_logic/      # Categorical and logical foundations
│   ├── logic/           # Knowledge representation
│   ├── tools/           # Agent tooling (bash, code, MCP)
│   └── tests/           # Test suite
├── autonomous-coding/    # Autonomous coding agent
├── trainer/             # Watson-Glaser style training UI
├── financial-data-analyst/  # Financial analysis demo
├── data/                # Schemas and reference data
└── tests/               # Additional test modules
```

## Architecture

The system implements an 8X architecture with these layers:

1. **Perception** - Input parsing and semantic analysis
2. **Memory** - Episodic and working memory management
3. **Reasoning** - Logic orchestration and inference
4. **Planning** - Goal decomposition and action selection
5. **Action** - Tool execution and output generation
6. **Reflection** - Self-evaluation and uncertainty estimation
7. **Safety** - Constraint checking and hallucination detection
8. **Observability** - Telemetry, tracing, and debugging

## Testing

```bash
# Run all tests
pytest

# Run specific test modules
pytest agents/tests/test_logic_orchestrator.py -v
pytest agents/tests/test_architectural_compliance.py -v
```

## License

MIT License - see LICENSE file for details.
