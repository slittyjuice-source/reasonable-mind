"""Edge case tests for governance components.

Tests for PlanValidator, ProcessGate, ConstraintRegistry, and ExecutionProxy
boundary conditions and validation constraints not covered elsewhere.
"""

from __future__ import annotations

import pytest

from agents.governance import (
    ConstraintRegistry,
    ExecutionContext,
    ExecutionMode,
    ExecutionProxy,
    ExecutionResult,
    Plan,
    PlanStep,
    PlanValidator,
    ProcessGate,
    ProcessStage,
    StageResult,
    ViolationType,
)

# =============================================================================
# PlanValidator Edge Cases
# =============================================================================


class TestPlanValidatorEdges:
    """Test PlanValidator boundary conditions and validation rules."""

    def test_load_plan_exceeds_max_steps(self, plan_validator: PlanValidator) -> None:
        """Reject plans that exceed max_steps limit."""
        plan = Plan(
            plan_id="too-big",
            steps=[PlanStep(id=f"s{i}", goal=f"Step {i}") for i in range(25)],
            constraint_profile="default",
            persona_id="test",
            max_steps=20,
        )
        result = plan_validator.load_plan(plan)
        assert not result.is_valid
        assert len(result.violations) == 1
        assert result.violations[0].violation_type == ViolationType.PLAN_TOO_LARGE

    def test_amend_plan_no_active_plan(self, plan_validator: PlanValidator) -> None:
        """Amending without an active plan fails with UNPLANNED_ACTION."""
        step = PlanStep(id="new", goal="New step")
        result = plan_validator.amend_plan(step)
        assert not result.is_valid
        assert result.violations[0].violation_type == ViolationType.UNPLANNED_ACTION

    def test_amend_plan_at_capacity(
        self, plan_validator: PlanValidator, full_plan: Plan
    ) -> None:
        """Amending a plan at max capacity fails with PLAN_TOO_LARGE."""
        plan_validator.load_plan(full_plan)
        step = PlanStep(id="overflow", goal="One too many")
        result = plan_validator.amend_plan(step)
        assert not result.is_valid
        assert result.violations[0].violation_type == ViolationType.PLAN_TOO_LARGE

    def test_validate_action_no_plan_loaded(
        self, plan_validator: PlanValidator
    ) -> None:
        """Validating action without loaded plan fails."""
        result = plan_validator.validate_action("read_file", {}, "step-1")
        assert not result.is_valid
        assert result.violations[0].violation_type == ViolationType.UNPLANNED_ACTION
        assert "No plan loaded" in result.violations[0].detail

    def test_validate_action_missing_citation(
        self, loaded_validator: PlanValidator
    ) -> None:
        """Validating action without plan_step_id fails with MISSING_PLAN_CITATION."""
        result = loaded_validator.validate_action("read_file", {}, plan_step_id=None)
        assert not result.is_valid
        assert (
            result.violations[0].violation_type == ViolationType.MISSING_PLAN_CITATION
        )

    def test_validate_action_step_not_found(
        self, loaded_validator: PlanValidator
    ) -> None:
        """Validating action with non-existent step fails."""
        result = loaded_validator.validate_action("read_file", {}, "nonexistent-step")
        assert not result.is_valid
        assert result.violations[0].violation_type == ViolationType.UNPLANNED_ACTION
        assert "Plan step not found" in result.violations[0].detail

    def test_validate_action_not_in_allowed(
        self, loaded_validator: PlanValidator
    ) -> None:
        """Validating action not in allowed_actions fails."""
        result = loaded_validator.validate_action("delete_file", {}, "step-1")
        assert not result.is_valid
        assert result.violations[0].violation_type == ViolationType.UNPLANNED_ACTION
        assert "Action not in allowed actions" in result.violations[0].detail

    def test_validate_destructive_action_unapproved(
        self, plan_validator: PlanValidator
    ) -> None:
        """Destructive actions require explicit approval."""
        plan = Plan(
            plan_id="cleanup",
            steps=[PlanStep(id="del", goal="Delete", allowed_actions=["delete_file"])],
            constraint_profile="default",
            persona_id="test",
        )
        plan_validator.load_plan(plan)
        result = plan_validator.validate_action("delete_file", {}, "del")
        assert not result.is_valid
        assert (
            result.violations[0].violation_type == ViolationType.DESTRUCTIVE_OPERATION
        )

    def test_validate_destructive_action_approved(
        self, plan_validator: PlanValidator
    ) -> None:
        """Destructive actions with approved=True pass validation."""
        plan = Plan(
            plan_id="cleanup",
            steps=[PlanStep(id="del", goal="Delete", allowed_actions=["delete_file"])],
            constraint_profile="default",
            persona_id="test",
        )
        plan_validator.load_plan(plan)
        result = plan_validator.validate_action(
            "delete_file", {"approved": True}, "del"
        )
        assert result.is_valid


# =============================================================================
# ProcessGate Edge Cases
# =============================================================================


class TestProcessGateEdges:
    """Test ProcessGate boundary conditions and sequencing rules."""

    def test_record_stage_out_of_order(self, process_gate: ProcessGate) -> None:
        """Recording stages out of order raises ValueError."""
        # Skip QUESTION_ANALYSIS and try to record CONCEPT_EXTRACTION
        result = StageResult(
            stage=ProcessStage.CONCEPT_EXTRACTION,
            passed=True,
            confidence=0.9,
        )
        with pytest.raises(ValueError, match="QUESTION_ANALYSIS not completed"):
            process_gate.record_stage(result)

    def test_record_stage_after_failure(self, process_gate: ProcessGate) -> None:
        """Recording stage after a failure raises ValueError."""
        # Record first stage as failed
        failed = StageResult(
            stage=ProcessStage.QUESTION_ANALYSIS,
            passed=False,
            confidence=0.3,
        )
        process_gate.record_stage(failed)

        # Try to record next stage
        next_stage = StageResult(
            stage=ProcessStage.CONCEPT_EXTRACTION,
            passed=True,
            confidence=0.9,
        )
        with pytest.raises(ValueError, match="QUESTION_ANALYSIS failed"):
            process_gate.record_stage(next_stage)

    def test_can_output_with_clarification_needed(
        self, process_gate: ProcessGate
    ) -> None:
        """Can't output if any stage needs clarification."""
        stages = list(ProcessStage)
        for i, stage in enumerate(stages):
            needs_clarification = "What do you mean?" if i == 3 else None
            result = StageResult(
                stage=stage,
                passed=True,
                confidence=0.9,
                needs_clarification=needs_clarification,
            )
            process_gate.record_stage(result)

        can_out, reason = process_gate.can_output()
        assert not can_out
        assert reason is not None and "needs clarification" in reason

    def test_can_output_with_low_confidence(self, process_gate: ProcessGate) -> None:
        """Can't output if any stage has confidence below threshold."""
        stages = list(ProcessStage)
        for i, stage in enumerate(stages):
            # Make one stage have low confidence
            confidence = 0.4 if i == 2 else 0.9
            result = StageResult(stage=stage, passed=True, confidence=confidence)
            process_gate.record_stage(result)

        can_out, reason = process_gate.can_output()
        assert not can_out
        assert reason is not None and "confidence below threshold" in reason

    def test_can_output_incomplete_stages(self, partial_gate: ProcessGate) -> None:
        """Can't output if not all stages completed."""
        can_out, reason = partial_gate.can_output()
        assert not can_out
        assert reason is not None and "Missing stages" in reason


# =============================================================================
# ConstraintRegistry Edge Cases
# =============================================================================


class TestConstraintRegistryEdges:
    """Test ConstraintRegistry boundary conditions and integrity verification."""

    def test_empty_registry_hash_is_none(
        self, constraint_registry: ConstraintRegistry
    ) -> None:
        """Empty registry has None as active_hash."""
        assert constraint_registry.active_hash is None

    def test_verify_integrity_empty_registry(
        self, constraint_registry: ConstraintRegistry
    ) -> None:
        """Integrity verification on empty registry fails for any hash."""
        assert not constraint_registry.verify_integrity("any-hash")

    def test_verify_integrity_wrong_hash(
        self, loaded_registry: ConstraintRegistry
    ) -> None:
        """Integrity verification fails for incorrect hash."""
        assert not loaded_registry.verify_integrity("wrong-hash")

    def test_verify_integrity_correct_hash(
        self, loaded_registry: ConstraintRegistry
    ) -> None:
        """Integrity verification passes for correct hash."""
        current_hash = loaded_registry.active_hash
        assert current_hash is not None
        assert loaded_registry.verify_integrity(current_hash)

    def test_hash_changes_on_profile_add(
        self, loaded_registry: ConstraintRegistry
    ) -> None:
        """Adding a new profile changes the active_hash."""
        original_hash = loaded_registry.active_hash
        loaded_registry.load_from_dict("second-profile", {"key": "value"})
        assert loaded_registry.active_hash != original_hash

    def test_clear_resets_hash(self, loaded_registry: ConstraintRegistry) -> None:
        """Clearing the registry resets active_hash to None."""
        assert loaded_registry.active_hash is not None
        loaded_registry.clear()
        assert loaded_registry.active_hash is None

    def test_hash_stable_on_key_order_change(
        self, constraint_registry: ConstraintRegistry
    ) -> None:
        """Hash is stable regardless of key order in data dict."""
        # Load with one order
        data1 = {"a": 1, "b": 2, "c": 3}
        constraint_registry.load_from_dict("profile1", data1)
        hash1 = constraint_registry.active_hash

        # Clear and reload with different key order
        constraint_registry.clear()
        data2 = {"c": 3, "b": 2, "a": 1}
        constraint_registry.load_from_dict("profile1", data2)
        hash2 = constraint_registry.active_hash

        # Hashes should be identical (canonicalization via sort_keys)
        assert hash1 == hash2


# =============================================================================
# ExecutionProxy Mock Mode Tests
# =============================================================================


class TestExecutionProxyMockMode:
    """Test MOCK mode - security boundary for testing."""

    def test_mock_message_fallback_to_stdout(self) -> None:
        """When mock has message but no stdout, message becomes stdout."""
        proxy = ExecutionProxy(mode=ExecutionMode.MOCK)
        mock_result = ExecutionResult(
            stdout="",
            stderr="",
            mode=ExecutionMode.MOCK,
            exit_code=0,
            message="test message",
        )
        proxy.register_mock(r"echo.*", mock_result)
        result = proxy.execute("echo hello")

        assert result.stdout == "test message"  # Fallback occurred

    def test_mock_with_stdout_uses_stdout(self) -> None:
        """When mock has stdout set, it is used directly."""
        proxy = ExecutionProxy(mode=ExecutionMode.MOCK)
        mock_result = ExecutionResult(
            stdout="explicit output",
            stderr="",
            mode=ExecutionMode.MOCK,
            exit_code=0,
            message="ignored message",
        )
        proxy.register_mock(r"echo.*", mock_result)
        result = proxy.execute("echo hello")

        assert result.stdout == "explicit output"

    def test_mock_no_match_returns_default(self) -> None:
        """When no mock matches, returns default result."""
        proxy = ExecutionProxy(mode=ExecutionMode.MOCK)
        proxy.register_mock(
            r"^specific$",
            ExecutionResult(
                stdout="won't match",
                stderr="",
                mode=ExecutionMode.MOCK,
                exit_code=0,
            ),
        )
        result = proxy.execute("echo different")

        assert result.mode == ExecutionMode.MOCK
        assert "[MOCK] Simulated execution" in result.stdout


# =============================================================================
# ExecutionContext Validation Tests
# =============================================================================


class TestExecutionContextValidation:
    """Test context validation - constraint binding integrity."""

    @pytest.mark.parametrize(
        "field,value",
        [
            ("constraint_hash", ""),
            ("plan_id", ""),
            ("persona_id", ""),
        ],
    )
    def test_validate_fails_on_empty_field(self, field: str, value: str) -> None:
        """Validation fails when any required field is empty."""
        kwargs = {
            "constraint_hash": "hash",
            "plan_id": "plan",
            "persona_id": "persona",
        }
        kwargs[field] = value
        context = ExecutionContext(**kwargs)
        assert context.validate() is False

    def test_validate_passes_with_all_fields(self) -> None:
        """Validation passes when all required fields are populated."""
        context = ExecutionContext(
            constraint_hash="hash123",
            plan_id="plan-001",
            persona_id="persona-A",
        )
        assert context.validate() is True


# =============================================================================
# ExecutionResult Property Tests
# =============================================================================


class TestExecutionResultProperties:
    """Test ExecutionResult property accessors when context is None."""

    def test_constraint_hash_without_context(self) -> None:
        """constraint_hash returns None when no context."""
        result = ExecutionResult(
            stdout="",
            stderr="",
            mode=ExecutionMode.LIVE,
            exit_code=0,
            execution_context=None,
        )
        assert result.constraint_hash is None

    def test_plan_id_without_context(self) -> None:
        """plan_id returns None when no context."""
        result = ExecutionResult(
            stdout="",
            stderr="",
            mode=ExecutionMode.LIVE,
            exit_code=0,
            execution_context=None,
        )
        assert result.plan_id is None

    def test_persona_id_without_context(self) -> None:
        """persona_id returns None when no context."""
        result = ExecutionResult(
            stdout="",
            stderr="",
            mode=ExecutionMode.LIVE,
            exit_code=0,
            execution_context=None,
        )
        assert result.persona_id is None

    def test_to_audit_record_without_context(self) -> None:
        """to_audit_record handles missing context gracefully."""
        result = ExecutionResult(
            stdout="output",
            stderr="",
            mode=ExecutionMode.DRY_RUN,
            exit_code=0,
            execution_context=None,
        )
        audit = result.to_audit_record()

        # All keys present with None values for context fields
        assert audit["constraint_hash"] is None
        assert audit["plan_id"] is None
        assert audit["persona_id"] is None
        assert audit["session_id"] is None
        assert audit["stdout"] == "output"
        assert "timestamp" in audit
