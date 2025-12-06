"""Domain-specific scenario bundles and prompts.

Each bundle pairs a tailored system prompt with recommended tools and a
multi-agent handoff sequence inspired by the autonomous-coding harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ScenarioBundle:
    """A reusable bundle for initializing an agent in a domain."""

    name: str
    description: str
    system_prompt: str
    tool_names: List[str]
    handoff_sequence: List[str] = field(default_factory=list)
    safety_posture: str = "logic-weighted with audit trail"


SCENARIO_LIBRARY: Dict[str, ScenarioBundle] = {
    "governance": ScenarioBundle(
        name="Governance",
        description="Policy-aware reasoning with procedural safeguards and auditability.",
        system_prompt=(
            "You are a governance advisor. Weight formal logic and precedent heavily, "
            "surface conflicts, and recommend procedural next steps with citations."
        ),
        tool_names=["web_search", "think", "calculator"],
        handoff_sequence=["policy-analysis", "risk-triage", "decision-draft"],
    ),
    "compliance": ScenarioBundle(
        name="Compliance",
        description="Control testing, obligation mapping, and evidence tracking.",
        system_prompt=(
            "You are a compliance analyst. Map requirements to controls, flag gaps, "
            "and propose remediation with traceable evidence and risk bands."
        ),
        tool_names=["file_tools", "web_search", "think"],
        handoff_sequence=["obligation-intake", "evidence-scan", "remediation-plan"],
        safety_posture="strict allowlist for file access and bash execution",
    ),
    "scientific_review": ScenarioBundle(
        name="Scientific Review",
        description="Peer-review style appraisal with replication checks and uncertainty quantification.",
        system_prompt=(
            "You are a scientific reviewer. Evaluate methodology, quantify uncertainty, "
            "and suggest replication strategies using conservative inference."
        ),
        tool_names=["calculator", "web_search", "think"],
        handoff_sequence=["claim-mapping", "evidence-weighing", "replication-plan"],
        safety_posture="logic-majority voting with counterfactual probes",
    ),
}


def get_scenario_bundle(name: str) -> ScenarioBundle:
    """Return a scenario bundle by key, raising for unknown entries."""

    key = name.lower()
    if key not in SCENARIO_LIBRARY:
        raise KeyError(f"Scenario '{name}' is not defined in the library")
    return SCENARIO_LIBRARY[key]

