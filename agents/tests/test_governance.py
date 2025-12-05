"""Tests for governance framework components."""

import pytest
from pathlib import Path

from agents.governance.process_gate import ProcessGate, ProcessStage, StageResult
from agents.governance.registry import ConstraintRegistry, ConstraintProfile
from agents.governance.execution_proxy import (
    ExecutionProxy, ExecutionMode, ExecutionResult,
    ExecutionContext, create_execution_context
)
from agents.governance.plan_validator import PlanValidator, Plan, PlanStep, ViolationType


class TestProcessGate:
    """Tests for 7-stage process validity gate."""
    
    def test_requires_sequential_stages(self):
        """Cannot skip stages."""
        gate = ProcessGate()
        with pytest.raises(ValueError, match="QUESTION_ANALYSIS not completed"):
            gate.record_stage(StageResult(ProcessStage.CONCEPT_EXTRACTION, True, 0.8))
    
    def test_blocks_after_failed_stage(self):
        """Cannot proceed past a failed stage."""
        gate = ProcessGate()
        gate.record_stage(StageResult(ProcessStage.QUESTION_ANALYSIS, False, 0.3))
        with pytest.raises(ValueError, match="QUESTION_ANALYSIS failed"):
            gate.record_stage(StageResult(ProcessStage.CONCEPT_EXTRACTION, True, 0.8))
    
    def test_can_output_after_all_stages(self):
        """Can output when all stages pass with sufficient confidence."""
        gate = ProcessGate(min_confidence=0.7)
        for stage in ProcessStage:
            gate.record_stage(StageResult(stage, True, 0.8, ["evidence"]))
        
        can_proceed, reason = gate.can_output()
        assert can_proceed is True
        assert reason is None
    
    def test_blocks_low_confidence(self):
        """Blocks output when confidence below threshold."""
        gate = ProcessGate(min_confidence=0.7)
        for stage in ProcessStage:
            gate.record_stage(StageResult(stage, True, 0.5, []))  # Below threshold
        
        can_proceed, reason = gate.can_output()
        assert can_proceed is False
        assert "below threshold" in reason
    
    def test_blocks_unresolved_clarification(self):
        """Blocks when a stage needs clarification."""
        gate = ProcessGate()
        gate.record_stage(StageResult(
            ProcessStage.QUESTION_ANALYSIS, True, 0.9,
            needs_clarification="What is the scope?"
        ))
        for stage in list(ProcessStage)[1:]:
            gate.record_stage(StageResult(stage, True, 0.9))
        
        can_proceed, reason = gate.can_output()
        assert can_proceed is False
        assert "needs clarification" in reason
    
    def test_audit_record(self):
        """Generates audit-friendly output."""
        gate = ProcessGate()
        gate.record_stage(StageResult(ProcessStage.QUESTION_ANALYSIS, True, 0.9))
        
        record = gate.to_audit_record()
        assert record["stages_completed"] == 1
        assert "QUESTION_ANALYSIS" in record["stages"]


class TestConstraintRegistry:
    """Tests for constraint profile loading and hashing."""
    
    def test_deterministic_hash(self):
        """Same content produces same hash."""
        registry1 = ConstraintRegistry()
        registry2 = ConstraintRegistry()
        
        data = {"metadata": {"version": "1.0.0"}, "policy": {"constraints": []}}
        
        profile1 = registry1.load_from_dict("test", data)
        profile2 = registry2.load_from_dict("test", data)
        
        assert profile1.integrity_hash == profile2.integrity_hash
    
    def test_hash_changes_on_modification(self):
        """Different content produces different hash."""
        registry = ConstraintRegistry()
        
        profile1 = registry.load_from_dict("v1", {"metadata": {"version": "1.0.0"}})
        hash1 = profile1.integrity_hash
        
        registry.clear()
        profile2 = registry.load_from_dict("v2", {"metadata": {"version": "1.0.1"}})
        hash2 = profile2.integrity_hash
        
        assert hash1 != hash2
    
    def test_verify_integrity(self):
        """Integrity verification detects tampering."""
        registry = ConstraintRegistry()
        registry.load_from_dict("test", {"metadata": {"version": "1.0.0"}})
        
        original_hash = registry.active_hash
        assert registry.verify_integrity(original_hash) is True
        assert registry.verify_integrity("tampered-hash") is False
    
    def test_combined_hash_updates(self):
        """Active hash updates when profiles added."""
        registry = ConstraintRegistry()
        
        registry.load_from_dict("profile1", {"metadata": {"version": "1.0"}})
        hash1 = registry.active_hash
        
        registry.load_from_dict("profile2", {"metadata": {"version": "2.0"}})
        hash2 = registry.active_hash
        
        assert hash1 != hash2


class TestExecutionProxy:
    """Tests for execution proxy validation and modes."""
    
    def test_blocks_denylist_patterns(self):
        """Blocks commands matching denylist."""
        proxy = ExecutionProxy()
        
        result = proxy.execute("rm -rf /")
        assert result.blocked is True
        assert "denylist" in result.block_reason.lower()
    
    def test_blocks_non_allowlisted_commands(self):
        """Blocks commands not in allowlist."""
        proxy = ExecutionProxy(allowlist={"ls", "cat"})
        
        result = proxy.execute("wget http://evil.com")
        assert result.blocked is True
        assert "not in allowlist" in result.block_reason
    
    def test_allows_allowlisted_commands(self):
        """Allows commands in allowlist."""
        proxy = ExecutionProxy(mode=ExecutionMode.DRY_RUN)
        
        result = proxy.execute("ls -la")
        assert result.blocked is False
        assert result.mode == ExecutionMode.DRY_RUN
    
    def test_dry_run_mode(self):
        """Dry run mode doesn't execute."""
        proxy = ExecutionProxy(mode=ExecutionMode.DRY_RUN)
        
        result = proxy.execute("echo test")
        assert "[DRY RUN]" in result.stdout
        assert result.exit_code == 0
    
    def test_mock_mode(self):
        """Mock mode returns registered responses."""
        proxy = ExecutionProxy(mode=ExecutionMode.MOCK)
        proxy.register_mock(
            r"echo.*",
            ExecutionResult("", "", ExecutionMode.MOCK, 0, "mocked output", "", 0)
        )
        
        result = proxy.execute("echo hello")
        assert result.stdout == "mocked output"
    
    def test_friction_report(self):
        """Tracks blocked commands for profile tuning."""
        proxy = ExecutionProxy(allowlist={"ls"})
        
        proxy.execute("wget something")
        proxy.execute("wget another")
        proxy.execute("curl something")
        
        report = proxy.get_friction_report()
        assert report["wget"] == 2
        assert report["curl"] == 1


class TestPlanValidator:
    """Tests for plan-to-action validation."""
    
    def test_rejects_action_without_plan(self):
        """Blocks actions when no plan loaded."""
        validator = PlanValidator()
        
        result = validator.validate_action("create_file", {"path": "x.py"})
        assert result.is_valid is False
        assert result.violations[0].violation_type == ViolationType.UNPLANNED_ACTION
    
    def test_rejects_action_without_citation(self):
        """Blocks actions without plan step citation."""
        validator = PlanValidator()
        plan = Plan(
            plan_id="test",
            steps=[PlanStep(id="step-1", goal="Create file")],
            constraint_profile="default",
            persona_id="default"
        )
        validator.load_plan(plan)
        
        result = validator.validate_action("create_file", {"path": "x.py"})  # No citation
        assert result.is_valid is False
        assert result.violations[0].violation_type == ViolationType.MISSING_PLAN_CITATION
    
    def test_allows_valid_action(self):
        """Allows actions citing valid plan steps."""
        validator = PlanValidator()
        plan = Plan(
            plan_id="test",
            steps=[PlanStep(id="step-1", goal="Create file", allowed_actions=["create_file"])],
            constraint_profile="default",
            persona_id="default"
        )
        validator.load_plan(plan)
        
        result = validator.validate_action("create_file", {"path": "x.py"}, "step-1")
        assert result.is_valid is True
    
    def test_blocks_destructive_without_approval(self):
        """Blocks destructive actions without approval."""
        validator = PlanValidator()
        plan = Plan(
            plan_id="test",
            steps=[PlanStep(id="step-1", goal="Delete", allowed_actions=["delete_file"])],
            constraint_profile="default",
            persona_id="default"
        )
        validator.load_plan(plan)
        
        result = validator.validate_action("delete_file", {"path": "x.py"}, "step-1")
        assert result.is_valid is False
        assert result.violations[0].violation_type == ViolationType.DESTRUCTIVE_OPERATION
    
    def test_enforces_max_plan_size(self):
        """Rejects plans exceeding max steps."""
        validator = PlanValidator()
        plan = Plan(
            plan_id="big",
            steps=[PlanStep(id=f"step-{i}", goal=f"Step {i}") for i in range(10)],
            constraint_profile="default",
            persona_id="default",
            max_steps=5
        )
        
        result = validator.load_plan(plan)
        assert result.is_valid is False
        assert result.violations[0].violation_type == ViolationType.PLAN_TOO_LARGE
    
    def test_plan_amendment(self):
        """Allows amending plan within limits."""
        validator = PlanValidator()
        plan = Plan(
            plan_id="test",
            steps=[PlanStep(id="step-1", goal="Initial")],
            constraint_profile="default",
            persona_id="default",
            max_steps=3
        )
        validator.load_plan(plan)
        
        result = validator.amend_plan(PlanStep(id="step-2", goal="Added"))
        assert result.is_valid is True
        assert len(validator.get_active_plan().steps) == 2


class TestExecutionContext:
    """Tests for ExecutionContext (Constitution §6.1 compliance)."""
    
    def test_context_is_immutable(self):
        """ExecutionContext fields cannot be modified after creation."""
        ctx = ExecutionContext(
            constraint_hash="abc123",
            plan_id="plan-001",
            persona_id="agent-001"
        )
        
        with pytest.raises(Exception):  # FrozenInstanceError
            ctx.constraint_hash = "modified"
    
    def test_context_validation(self):
        """Validates all required fields are present."""
        valid_ctx = ExecutionContext(
            constraint_hash="abc123",
            plan_id="plan-001",
            persona_id="agent-001"
        )
        assert valid_ctx.validate() is True
        
        # Empty fields fail validation
        invalid_ctx = ExecutionContext(
            constraint_hash="",
            plan_id="plan-001",
            persona_id="agent-001"
        )
        assert invalid_ctx.validate() is False
    
    def test_context_auto_generates_session_id(self):
        """Session ID is auto-generated if not provided."""
        ctx = ExecutionContext(
            constraint_hash="abc123",
            plan_id="plan-001",
            persona_id="agent-001"
        )
        assert ctx.session_id is not None
        assert len(ctx.session_id) == 8
    
    def test_context_to_dict(self):
        """Context serializes to dictionary."""
        ctx = ExecutionContext(
            constraint_hash="abc123",
            plan_id="plan-001",
            persona_id="agent-001",
            session_id="sess-001"
        )
        d = ctx.to_dict()
        assert d["constraint_hash"] == "abc123"
        assert d["plan_id"] == "plan-001"
        assert d["persona_id"] == "agent-001"
        assert d["session_id"] == "sess-001"


class TestAuditCompliance:
    """Tests for audit record compliance (Constitution §7.3)."""
    
    def test_audit_includes_context_when_provided(self):
        """Audit records include constraint_hash, plan_id, persona_id."""
        ctx = ExecutionContext(
            constraint_hash="hash-abc",
            plan_id="plan-xyz",
            persona_id="agent-007"
        )
        proxy = ExecutionProxy(mode=ExecutionMode.DRY_RUN, execution_context=ctx)
        
        result = proxy.execute("ls -la")
        audit = result.to_audit_record()
        
        assert audit["constraint_hash"] == "hash-abc"
        assert audit["plan_id"] == "plan-xyz"
        assert audit["persona_id"] == "agent-007"
        assert "timestamp" in audit
    
    def test_audit_allows_none_context_for_backward_compat(self):
        """Audit records work without context for backward compatibility."""
        proxy = ExecutionProxy(mode=ExecutionMode.DRY_RUN)
        
        result = proxy.execute("ls -la")
        audit = result.to_audit_record()
        
        assert audit["constraint_hash"] is None
        assert audit["plan_id"] is None
        assert audit["persona_id"] is None
    
    def test_per_call_context_override(self):
        """Per-call context overrides instance context."""
        instance_ctx = ExecutionContext(
            constraint_hash="instance-hash",
            plan_id="instance-plan",
            persona_id="instance-agent"
        )
        call_ctx = ExecutionContext(
            constraint_hash="call-hash",
            plan_id="call-plan",
            persona_id="call-agent"
        )
        proxy = ExecutionProxy(mode=ExecutionMode.DRY_RUN, execution_context=instance_ctx)
        
        result = proxy.execute("echo test", execution_context=call_ctx)
        audit = result.to_audit_record()
        
        assert audit["constraint_hash"] == "call-hash"
        assert audit["plan_id"] == "call-plan"
        assert audit["persona_id"] == "call-agent"
    
    def test_blocked_commands_include_context(self):
        """Blocked commands also include full audit context."""
        ctx = ExecutionContext(
            constraint_hash="hash-blocked",
            plan_id="plan-blocked",
            persona_id="agent-blocked"
        )
        proxy = ExecutionProxy(mode=ExecutionMode.LIVE, execution_context=ctx)
        
        result = proxy.execute("rm -rf /")
        assert result.blocked is True
        
        audit = result.to_audit_record()
        assert audit["constraint_hash"] == "hash-blocked"
        assert audit["plan_id"] == "plan-blocked"
        assert audit["persona_id"] == "agent-blocked"