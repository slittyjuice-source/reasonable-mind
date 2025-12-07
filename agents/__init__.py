"""Core agent implementations."""

# Lazy imports keep optional dependencies (like the anthropic SDK) from being
# required at package import time. This allows subpackages such as
# ``agents.governance`` to be imported in environments where the SDK is not
# installed, while still providing the same public API when users access
# ``Agent`` or related classes directly.

__all__ = ["Agent", "ModelConfig", "Tool"]


def __getattr__(name):
    if name in {"Agent", "ModelConfig"}:
        from .agent import Agent, ModelConfig

        return {"Agent": Agent, "ModelConfig": ModelConfig}[name]
    if name == "Tool":
        from .tools.base import Tool

        return Tool
    raise AttributeError(f"module 'agents' has no attribute '{name}'")
