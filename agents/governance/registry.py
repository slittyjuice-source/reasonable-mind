"""
Constraint profile loader with cryptographic integrity verification.

Loads YAML/JSON constraint profiles, canonicalizes them, and computes
SHA-256 hashes for tamper detection. Active hash is emitted with every
plan/action log for audit verification.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _utc_now() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass
class ConstraintProfile:
    """Loaded and validated constraint profile with integrity hash."""
    profile_id: str
    version: str
    constraints: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    integrity_hash: str
    loaded_at: datetime = field(default_factory=_utc_now)
    
    def to_audit_record(self) -> Dict[str, Any]:
        """Generate audit-friendly representation."""
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "integrity_hash": self.integrity_hash,
            "constraint_count": len(self.constraints),
            "loaded_at": self.loaded_at.isoformat()
        }


class ConstraintRegistry:
    """
    Loads constraint profiles with schema validation and cryptographic verification.
    
    Usage:
        registry = ConstraintRegistry(Path("policies/"))
        profile = registry.load_profile(Path("policies/security.yaml"))
        print(registry.active_hash)  # Emit with every log
    """
    
    def __init__(self, policies_dir: Optional[Path] = None):
        self.policies_dir = policies_dir
        self._active_profiles: Dict[str, ConstraintProfile] = {}
        self._active_hash: Optional[str] = None
    
    def _canonicalize(self, data: Dict[str, Any]) -> str:
        """Canonicalize payload for deterministic hashing."""
        return json.dumps(data, sort_keys=True, separators=(',', ':'))
    
    def _compute_hash(self, canonical: str) -> str:
        """Compute SHA-256 hash of canonicalized payload."""
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()
    
    def load_profile(self, profile_path: Path) -> ConstraintProfile:
        """
        Load a constraint profile with validation and hashing.
        
        Supports YAML (.yaml, .yml) and JSON (.json) files.
        """
        suffix = profile_path.suffix.lower()
        
        with open(profile_path) as f:
            if suffix in ('.yaml', '.yml'):
                try:
                    import yaml
                    raw_data = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML required for YAML files: pip install pyyaml")
            else:
                raw_data = json.load(f)
        
        # Extract components
        metadata = raw_data.get('metadata', {})
        policy = raw_data.get('policy', {})
        constraints = policy.get('constraints', [])
        
        # Canonicalize and hash
        canonical = self._canonicalize(raw_data)
        integrity_hash = self._compute_hash(canonical)
        
        profile = ConstraintProfile(
            profile_id=profile_path.stem,
            version=metadata.get('version', 'unknown'),
            constraints=constraints,
            metadata=metadata,
            integrity_hash=integrity_hash
        )
        
        self._active_profiles[profile.profile_id] = profile
        self._recompute_active_hash()
        
        return profile
    
    def load_from_dict(self, profile_id: str, data: Dict[str, Any]) -> ConstraintProfile:
        """Load a constraint profile from a dictionary (for testing)."""
        metadata = data.get('metadata', {})
        policy = data.get('policy', {})
        constraints = policy.get('constraints', [])
        
        canonical = self._canonicalize(data)
        integrity_hash = self._compute_hash(canonical)
        
        profile = ConstraintProfile(
            profile_id=profile_id,
            version=metadata.get('version', 'unknown'),
            constraints=constraints,
            metadata=metadata,
            integrity_hash=integrity_hash
        )
        
        self._active_profiles[profile_id] = profile
        self._recompute_active_hash()
        
        return profile
    
    def _recompute_active_hash(self) -> None:
        """Recompute combined hash of all active profiles."""
        combined = {
            pid: p.integrity_hash 
            for pid, p in sorted(self._active_profiles.items())
        }
        self._active_hash = self._compute_hash(self._canonicalize(combined))
    
    @property
    def active_hash(self) -> str:
        """Get the active constraint hash for audit logs."""
        return self._active_hash or "no-constraints-loaded"
    
    def verify_integrity(self, expected_hash: str) -> bool:
        """Verify current constraints match expected hash."""
        return self._active_hash == expected_hash
    
    def get_profile(self, profile_id: str) -> Optional[ConstraintProfile]:
        """Get a loaded profile by ID."""
        return self._active_profiles.get(profile_id)
    
    def get_all_constraints(self) -> List[Dict[str, Any]]:
        """Get all constraints from all loaded profiles."""
        all_constraints = []
        for profile in self._active_profiles.values():
            all_constraints.extend(profile.constraints)
        return all_constraints
    
    def check_constraint(self, constraint_id: str, context: Dict[str, Any]) -> bool:
        """
        Check if a specific constraint is satisfied.
        
        Returns True if constraint passes or doesn't exist.
        """
        for profile in self._active_profiles.values():
            for constraint in profile.constraints:
                if constraint.get('id') == constraint_id:
                    # Simple condition checking - extend as needed
                    condition = constraint.get('condition', {})
                    if condition.get('type') == 'command_pattern':
                        import re
                        command = context.get('command', '')
                        for pattern in condition.get('deny_patterns', []):
                            if re.search(pattern, command, re.IGNORECASE):
                                return False
                    return True
        return True  # Constraint not found = passes
    
    def clear(self) -> None:
        """Clear all loaded profiles."""
        self._active_profiles.clear()
        self._active_hash = None
