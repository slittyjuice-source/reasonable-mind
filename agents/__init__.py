"""Core agent implementations."""

# Lazy imports keep optional dependencies (like the anthropic SDK) from being
# required at package import time. This allows subpackages such as
# ``agents.governance`` to be imported in environments where the SDK is not
# installed, while still providing the same public API when users access
# ``Agent`` or related classes directly.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # These imports are only for static analysis and won't run at import time,
    # preserving the lazy-loading behavior for optional dependencies.
    from .agent import Agent, ModelConfig
    from .tools.base import Tool

__all__ = ["Agent", "ModelConfig", "Tool"]


def __getattr__(name):
    if name in {"Agent", "ModelConfig"}:
        from .agent import Agent, ModelConfig

        return {"Agent": Agent, "ModelConfig": ModelConfig}[name]
    if name == "Tool":
        from .tools.base import Tool

        return Tool
    raise AttributeError(f"module 'agents' has no attribute '{name}'")
