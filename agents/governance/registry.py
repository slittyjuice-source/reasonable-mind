"""Constraint registry with deterministic hashing."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Optional


def _hash_dict(data: dict) -> str:
    """Return deterministic SHA-256 hash of a dict."""

    normalized = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode()).hexdigest()


@dataclass
class ConstraintProfile:
    name: str
    data: dict
    integrity_hash: str


class ConstraintRegistry:
    """Loads and verifies constraint profiles."""

    def __init__(self) -> None:
        self._profiles: Dict[str, ConstraintProfile] = {}
        self.active_hash: Optional[str] = None

    def clear(self) -> None:
        self._profiles.clear()
        self.active_hash = None

    def _recalculate_active_hash(self) -> None:
        hashes = [p.integrity_hash for p in self._profiles.values()]
        combined = hashlib.sha256("".join(sorted(hashes)).encode()).hexdigest() if hashes else None
        self.active_hash = combined

    def load_from_dict(self, name: str, data: dict) -> ConstraintProfile:
        profile = ConstraintProfile(name=name, data=data, integrity_hash=_hash_dict(data))
        self._profiles[name] = profile
        self._recalculate_active_hash()
        return profile

    def verify_integrity(self, integrity_hash: str) -> bool:
        return integrity_hash == self.active_hash
