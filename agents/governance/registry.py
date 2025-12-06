"""
Constraint registry with hashing helpers for integrity checks.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Optional


def _stable_hash(data: object) -> str:
    """Compute a deterministic SHA-256 hash for arbitrary data."""

    normalized = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass
class ConstraintProfile:
    """Loaded constraint profile with integrity hash."""

    name: str
    data: Dict
    integrity_hash: str


class ConstraintRegistry:
    """Minimal registry that can load profiles and track integrity."""

    def __init__(self) -> None:
        self._profiles: Dict[str, ConstraintProfile] = {}
        self.active_hash: Optional[str] = None

    def _recompute_active_hash(self) -> None:
        """Hash of all loaded profiles to detect tampering."""

        if not self._profiles:
            self.active_hash = None
            return

        hashes = sorted(profile.integrity_hash for profile in self._profiles.values())
        combined = _stable_hash(hashes)
        self.active_hash = combined

    def load_from_dict(self, name: str, data: Dict) -> ConstraintProfile:
        """Load a profile from a dictionary and track integrity."""

        integrity_hash = _stable_hash(data)
        profile = ConstraintProfile(name=name, data=data, integrity_hash=integrity_hash)
        self._profiles[name] = profile
        self._recompute_active_hash()
        return profile

    def verify_integrity(self, integrity_hash: str) -> bool:
        """Verify the combined active hash matches the expected value."""

        return self.active_hash == integrity_hash

    def clear(self) -> None:
        """Reset registry state."""

        self._profiles.clear()
        self.active_hash = None
