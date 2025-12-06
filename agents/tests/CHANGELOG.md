# Changelog - agents/tests

All notable changes to the tests module will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **test_shell_injection_prevention.py**: Comprehensive security test suite for ExecutionProxy
  - 29 test cases covering all 18 denylist patterns (100% coverage)
  - 24 positive tests (blocking malicious commands)
  - 5 negative tests (allowing safe commands)
  - Command chaining tests: `;`, `|`, `&&`, `||`, `&` operators
  - Command substitution tests: `$()`, backtick patterns
  - Output redirection tests: `/etc/`, `/bin/`, `/usr/` protection
  - Fork bomb pattern detection
  - Mail/sendmail piping prevention
  - Direct pipe-to-shell blocking
  - Audit trail validation (Constitution §7.3 compliance)
  - Defense-in-depth testing (PlanValidator + ExecutionProxy)
  - Friction report tracking for policy tuning
  - All three execution modes (LIVE, DRY_RUN, MOCK)
  - Edge case handling (empty commands, whitespace)
  - Mock mode audit context propagation (regression test)

### Security

- **test_shell_injection_prevention.py**: Validates shell injection prevention patches
  - Verifies 8 new denylist patterns added in governance/execution_proxy.py
  - Tests command chaining prevention: `[;&|]\s*rm\b`, `[;&|]\s*curl\b`, `[;&|]\s*wget\b`
  - Tests command substitution blocking: `\$\(`, backtick
  - Tests output redirection protection: `>\s*/etc/`, `>\s*/bin/`, `>\s*/usr/`
  - Tests piping prevention: `[|]\s*mail\b`, `[|]\s*sendmail\b`, `[|]\s*(ba)?sh\b`
  - Tests fork bomb detection: `:[(][)][{].*[|].*&.*[}];:`
  - Regression test for Constitution §7.3: audit context in mock mode
  - Defense in depth: ExecutionProxy blocks despite PlanValidator authorization

## [1.0.0] - 2024-XX-XX

### Added

- Initial test suite for categorical_engine, logic_orchestrator, governance components
- Comprehensive test coverage for core reasoning modules
