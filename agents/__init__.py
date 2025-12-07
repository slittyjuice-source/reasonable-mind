"""Core agent implementations."""

from importlib import import_module

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


def __getattr__(name: str):
    """Lazily import heavy or optional dependencies.

    The :mod:`anthropic` package used by :class:`Agent` is not installed in all
    environments. Importing it at module load time caused pytest discovery to
    fail before tests even ran. By deferring the import we allow the rest of the
    package (e.g., governance utilities) to be used without the optional client
    dependency. If a caller explicitly accesses ``Agent`` or ``ModelConfig``
    without ``anthropic`` available, we surface a clear error message.
    """

    if name in {"Agent", "ModelConfig"}:
        try:
            module = import_module(".agent", __name__)
        except ModuleNotFoundError as exc:
            if exc.name == "anthropic":
                raise ImportError(
                    "The 'anthropic' package is required to use agents.Agent. "
                    "Install it or provide a compatible client instance.",
                ) from exc
            raise

        Agent = getattr(module, "Agent")
        ModelConfig = getattr(module, "ModelConfig")
        globals().update({"Agent": Agent, "ModelConfig": ModelConfig})
        return globals()[name]

    if name == "Tool":
        return Tool

    raise AttributeError(f"module 'agents' has no attribute '{name}'")
