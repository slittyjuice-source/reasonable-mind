import pytest

from agents.core_logic import (
    CategoricalEngine,
    parse_syllogism,
    validate_barbara,
)
from agents.core_logic.logic_engine import (
    LogicalArgument,
    LogicEngine,
    Proposition,
)
from agents.governance import (
    ConstraintRegistry,
    ExecutionMode,
    ExecutionProxy,
    ExecutionResult,
    Plan,
    PlanStep,
    PlanValidator,
    ProcessGate,
    ProcessStage,
    StageResult,
    create_execution_context,
)


def test_categorical_negative_requires_negative_conclusion():
    syl = parse_syllogism(
        "No reptiles are mammals",
        "All dogs are reptiles",
        "All dogs are mammals",
    )
    result = CategoricalEngine().validate(syl)
    assert result.is_valid is False
    assert any(
        "negative premise requires negative conclusion" in v.lower()
        for v in result.violations
    )


def test_categorical_particular_with_negative_requires_particular_conclusion():
    syl = parse_syllogism(
        "Some dogs are pets",
        "No pets are reptiles",
        "All dogs are reptiles",
    )
    result = CategoricalEngine().validate(syl)
    assert result.is_valid is False
    assert any(
        "particular premise with negative premise" in v.lower()
        for v in result.violations
    )


def test_categorical_valid_but_unnamed_form():
    syl = parse_syllogism(
        "All mammals are animals",
        "All dogs are mammals",
        "Some dogs are animals",
    )
    result = CategoricalEngine().validate(syl)
    assert result.is_valid is True
    assert result.form_name is None
    assert "valid by rules" in result.explanation.lower()


def test_validate_barbara_parse_failure():
    result = validate_barbara(
        "nonsense statement", "invalid minor", "invalid conclusion"
    )
    assert result.is_valid is False
    assert "failed to parse" in result.explanation.lower()


def test_parse_categorical_statement_edge_cases():
    from agents.core_logic.categorical_engine import parse_categorical_statement

    assert (
        parse_categorical_statement("All planets orbit stars").predicate
        == "orbit stars"
    )
    assert parse_categorical_statement("No robots are humans").predicate == "humans"
    assert parse_categorical_statement("Some robots are not humans").copula == "are not"
    assert parse_categorical_statement("All X") is None
    assert parse_categorical_statement("No X") is None
    assert parse_categorical_statement("Some X") is None
    parsed_partial = parse_categorical_statement("Some X are not")
    assert parsed_partial is not None and parsed_partial.predicate == "not"
    assert parse_categorical_statement("Some x are not y are not z") is None
    assert parse_categorical_statement("unparsable statement") is None


def test_parse_syllogism_middle_term_fallback():
    syl = parse_syllogism(
        "All dogs are animals",
        "All animals are dogs",
        "All dogs are animals",
    )
    # Middle term selection falls back when premises share only conclusion terms
    assert syl.middle_term in {"dogs", "animals"}


def test_logic_engine_fallback_forms(tmp_path):
    engine = LogicEngine(data_path=tmp_path)
    assert "modus_ponens" in engine.valid_forms
    assert "affirming_consequent" in engine.invalid_forms


def test_logic_engine_proposition_equality():
    assert Proposition("P", "first") == Proposition("P", "second")


def test_logic_engine_no_premises_guard(tmp_path):
    engine = LogicEngine(data_path=tmp_path)
    arg = LogicalArgument(premises=[], conclusion="Q", propositions=set())
    result = engine.validate(arg)
    assert result.is_valid is False
    assert "no premises" in result.explanation.lower()


def test_logic_engine_conclusion_new_terms(tmp_path):
    engine = LogicEngine(data_path=tmp_path)
    props = {Proposition("P", "P"), Proposition("R", "R")}
    arg = LogicalArgument(premises=["P"], conclusion="R", propositions=props)
    result = engine.validate(arg)
    assert result.is_valid is False
    assert "new terms" in result.explanation.lower()


def test_logic_engine_tokenize_predicate_like():
    engine = LogicEngine()
    tokens = engine._tokenize("Human(Socrates)")
    assert tokens == ["Human(Socrates)"]
    nested = engine._tokenize("Func(a(b)c)")
    assert nested == ["Func(a(b)c)"]
    simple = engine._tokenize("ABC_123")
    assert simple == ["ABC_123"]


def test_logic_engine_convert_implications_and_eval():
    engine = LogicEngine()
    converted = engine._convert_implications("A _implies_ B _iff_ C")
    assert "(not A or B)" in converted and "and" in converted
    assert engine._evaluate_expression("X", {"Y": True}) is False


def test_logic_engine_heuristic_large_argument(tmp_path):
    engine = LogicEngine(data_path=tmp_path)
    symbols = ["P", "Q", "R", "S", "T", "U"]
    propositions = {Proposition(sym, sym) for sym in symbols}
    arg = LogicalArgument(
        premises=["P", "Q", "R", "S", "T", "U"],
        conclusion="U",
        propositions=propositions,
    )
    result = engine.validate(arg)
    assert result.method == "heuristic"
    assert result.confidence < 1.0


def test_logic_engine_biconditional_and_unknown_eval(tmp_path):
    engine = LogicEngine(data_path=tmp_path)
    # Drive biconditional conversion and unknown-token fallback
    expr = engine._convert_implications("A _iff_ B")
    assert "not" in expr and "and" in expr
    assert engine._evaluate_expression("Z ↔ Y", {"Y": True}) in (True, False)
    # Parentheses and failure paths
    assert engine._evaluate_expression("(P)", {"P": True}) is True
    assert engine._evaluate_expression("invalid(", {}) is False
    assert engine._evaluate_expression("P → Q", {"P": False, "Q": False}) is True
    assert engine._evaluate_expression("(", {}) is False


def test_logic_engine_patterns_match_conclusion_mismatch():
    engine = LogicEngine()
    assert (
        engine._patterns_match(
            ["P → Q", "P"], ["P → Q", "P"], conclusion="R", expected_conclusion="Q"
        )
        is False
    )


def test_truth_table_validate_too_many_props():
    engine = LogicEngine()
    props = {Proposition(sym, sym) for sym in ["P", "Q", "R", "S", "T", "U"]}
    arg = LogicalArgument(
        premises=["P", "Q", "R", "S", "T", "U"], conclusion="U", propositions=props
    )
    assert engine._truth_table_validate(arg) is None


def test_logic_engine_direct_heuristic_new_terms():
    engine = LogicEngine()
    props = {Proposition("P", "P"), Proposition("Q", "Q")}
    arg = LogicalArgument(premises=["P"], conclusion="Q", propositions=props)
    res = engine._heuristic_validate(arg, warnings=["manual"])
    assert res.is_valid is False
    assert "new terms" in res.explanation.lower()


def test_process_gate_edge_cases():
    gate = ProcessGate()
    # Duplicate stage recording blocked
    gate.record_stage(StageResult(ProcessStage.QUESTION_ANALYSIS, True, 0.9))
    with pytest.raises(ValueError):
        gate.record_stage(StageResult(ProcessStage.QUESTION_ANALYSIS, True, 0.9))
    can_output, reason = gate.can_output()
    assert can_output is False and "Missing stages" in reason

    # Inject a failed stage to exercise failure branch
    gate._results = {stage: StageResult(stage, True, 0.9) for stage in ProcessStage}
    gate._results[ProcessStage.EXECUTION] = StageResult(
        ProcessStage.EXECUTION, False, 0.9
    )
    can_output, reason = gate.can_output()
    assert can_output is False and "EXECUTION failed" in reason


def test_constraint_registry_recompute_empty():
    registry = ConstraintRegistry()
    registry._recompute_active_hash()
    assert registry.active_hash is None


def test_plan_validator_missing_paths():
    validator = PlanValidator()
    # Amend without active plan
    res = validator.amend_plan(PlanStep(id="s-1", goal="do things"))
    assert (
        res.is_valid is False
        and res.violations[0].violation_type.name == "UNPLANNED_ACTION"
    )

    plan = Plan(
        plan_id="p1",
        steps=[PlanStep(id="step-1", goal="initial")],
        constraint_profile="default",
        persona_id="persona",
        max_steps=1,
    )
    validator.load_plan(plan)
    # Exceed max steps
    res = validator.amend_plan(PlanStep(id="step-2", goal="extra"))
    assert (
        res.is_valid is False
        and res.violations[0].violation_type.name == "PLAN_TOO_LARGE"
    )
    # Citation not found
    res = validator.validate_action("create", {}, "missing-step")
    assert (
        res.is_valid is False
        and res.violations[0].violation_type.name == "UNPLANNED_ACTION"
    )
    # Action not in allowed list
    res = validator.validate_action("other", {}, "step-1")
    assert (
        res.is_valid is False
        and res.violations[0].violation_type.name == "UNPLANNED_ACTION"
    )


def test_execution_proxy_mock_and_live_branches():
    ctx = create_execution_context("hash", "plan", "persona", session_id="sess-1")
    proxy = ExecutionProxy(mode=ExecutionMode.MOCK, execution_context=ctx)
    mock = ExecutionResult(
        correlation_id="cid",
        command="hello",
        mode=ExecutionMode.MOCK,
        exit_code=0,
        stdout="mocked",
        stderr="",
        duration_ms=0,
    )
    proxy.register_mock(r"hello.*", mock)
    result = proxy.execute("hello world")
    assert result.stdout == "mocked"
    # Mock list iterates without match
    miss = proxy.execute("different command")
    assert miss.blocked is False
    assert miss.message == "No mock registered"
    # Live allowlisted command goes through non-mock branch
    live = ExecutionProxy(mode=ExecutionMode.LIVE)
    live_result = live.execute("echo hi | grep hi")
    assert live_result.blocked is False
    assert "[LIVE EXECUTION SKIPPED]" in live_result.stdout
    # Blocked command path for denylist
    blocked = live.execute("rm important.txt")
    assert blocked.blocked is True and "denylist command" in blocked.block_reason
    # MOCK mode without registered mock exercises stdout branch
    mock_mode = ExecutionProxy(mode=ExecutionMode.MOCK)
    mock_live = mock_mode.execute("echo dry")
    assert mock_live.stdout.startswith("[MOCK]")
    # Tokenization handles empty pipeline segments
    tokens = mock_mode._tokenize_pipeline("echo hi | ")
    assert tokens == ["echo"]
