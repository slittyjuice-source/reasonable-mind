# Agent Governance Guide

A Reasonable Mind only proceeds when:

- **Process Validity** – All mandated reasoning stages completed
- **Governance Alignment** – Actions verified against active policy hash
- **Confidence Threshold** – Evidence-weighted consensus reached

---

## Quick Start

```python
from agents.governance import (
    ProcessGate,
    ConstraintRegistry,
    ExecutionProxy,
    PlanValidator,
    Plan,
    PlanStep,
)

# 1. Enforce reasoning pipeline
gate = ProcessGate(confidence_threshold=0.7)

# Record progress through stages
gate.record_stage("question_analysis", confidence=0.9, evidence=["parsed user intent"])
gate.record_stage("concept_identification", confidence=0.85, evidence=["identified 3 concepts"])
# ... complete all 7 stages

if gate.can_output():
    # Proceed with action
    pass
else:
    # Request clarification
    missing = gate.missing_stages()
```

## Components

### ProcessGate: 7-Stage Pipeline

Enforces the structured reasoning sequence:

| Stage | Purpose |
|-------|---------|
| question_analysis | Parse and understand user intent |
| concept_identification | Identify relevant concepts |
| layer_determination | Determine abstraction layers |
| strategy_selection | Select reasoning strategies |
| module_integration | Integrate relevant modules |
| evaluation | Evaluate options |
| consensus | Weighted consensus building |

```python
gate = ProcessGate(confidence_threshold=0.7)
gate.record_stage("question_analysis", confidence=0.9, evidence=["user wants X"])

# Check status
summary = gate.get_stage_summary()
# {'completed_count': 1, 'total_count': 7, 'can_output': False, ...}
```

### ConstraintRegistry: Policy Hashing

Loads constraint profiles with SHA-256 integrity verification:

```python
from pathlib import Path

registry = ConstraintRegistry(Path("agents/governance/policies/"))
profile = registry.load_profile(Path("agents/governance/policies/security.yaml"))

# Emit with every log for audit
print(f"Active hash: {registry.active_hash}")

# Detect tampering
if registry.verify_integrity():
    # Profiles unchanged since load
    pass
```

### ExecutionProxy: Shell Mediation

Single point of control for external commands:

```python
proxy = ExecutionProxy(mode=ExecutionMode.LIVE)

# Validate before execution
result = proxy.validate_command("ls -la")
if result.allowed:
    result = proxy.execute("ls -la")
    print(result.stdout)
else:
    print(f"Denied: {result.denied_reason}")

# Get friction report for tuning allowlist
report = proxy.get_friction_report()
```

Default allowlist: `ls`, `cat`, `head`, `tail`, `wc`, `grep`, `find`, `pwd`, `echo`, `date`

### PlanValidator: Plan-to-Action Mapping

Validates plans with contingencies and escape hatches:

```python
plan = Plan(
    goal="Refactor module X",
    steps=[
        PlanStep(
            action="Run tests",
            rationale="Establish baseline",
            contingencies={"tests_fail": "Fix before proceeding"},
            open_questions=["Which test suite?"]
        ),
        PlanStep(
            action="Apply refactoring",
            rationale="Improve structure",
            requires_approval=True  # Blocks until approved
        )
    ]
)

validator = PlanValidator()
result = validator.validate(plan)

if result.is_valid:
    # Execute plan
    pass
else:
    for v in result.violations:
        print(f"{v.violation_type}: {v.message}")
```

## Policy Files

### security.yaml

```yaml
policy:
  constraints:
    - id: no_destructive_ops
      rule: "Block rm -rf, shutdown, reboot"
      severity: critical
    - id: network_access
      rule: "Log all network operations"
      severity: high
```

### plan_amendment.yaml

```yaml
policy:
  constraints:
    - id: user_changes
      rule: "User edits take precedence, re-validate"
      severity: medium
    - id: blocker_contingency
      rule: "Blocked steps trigger contingency lookup"
      severity: high
```

## Integration Pattern

```python
class GovernedAgent:
    def __init__(self):
        self.gate = ProcessGate()
        self.registry = ConstraintRegistry(Path("policies/"))
        self.proxy = ExecutionProxy()
        self.validator = PlanValidator()
        
        # Load policies
        self.registry.load_profile(Path("policies/security.yaml"))
    
    def process(self, user_input: str) -> str:
        # 1. Work through reasoning stages
        self.gate.record_stage("question_analysis", 
                               confidence=0.9, 
                               evidence=[f"User asked: {user_input}"])
        # ... complete remaining stages
        
        # 2. Check process validity
        if not self.gate.can_output():
            return f"Need clarification on: {self.gate.missing_stages()}"
        
        # 3. Build and validate plan
        plan = self._build_plan()
        result = self.validator.validate(plan)
        
        if not result.is_valid:
            return f"Plan rejected: {result.violations}"
        
        # 4. Execute with governance
        for step in plan.steps:
            if step.requires_approval:
                # Request user approval
                pass
            
            # Execute through proxy
            if "shell:" in step.action:
                cmd = step.action.replace("shell:", "")
                exec_result = self.proxy.execute(cmd)
        
        # 5. Log with active hash for audit
        return f"Completed. Policy hash: {self.registry.active_hash}"
```

## Design Principles

1. **Govern, Observe, Adjust** – Start with minimal constraints, log friction, tune based on data
2. **Small Plans** – 2-5 steps max, treat as contracts with escape hatches
3. **Evidence-Weighted** – Confidence thresholds prevent premature conclusions
4. **Cryptographic Audit** – SHA-256 hashes prove policy state at decision time
5. **Friction Reports** – `ExecutionProxy.get_friction_report()` reveals over-constraining

## Extending

Add new policies in `agents/governance/policies/`:

```yaml
metadata:
  version: "1.0.0"
  description: "Your policy"
  
policy:
  constraints:
    - id: your_rule
      rule: "Description of constraint"
      severity: critical|high|medium|low
```

Load with: `registry.load_profile(Path("policies/your_policy.yaml"))`
