import os

from agents.tools.security import CommandSecurityPolicy, ValidationResult


def test_allowlisted_command_passes_validation():
    policy = CommandSecurityPolicy()

    result = policy.validate("ls -la")

    assert isinstance(result, ValidationResult)
    assert result.allowed


def test_block_non_allowlisted_command():
    policy = CommandSecurityPolicy()

    result = policy.validate("rm -rf /")

    assert not result.allowed
    assert "not allowed" in result.reason


def test_block_pkill_for_disallowed_process():
    policy = CommandSecurityPolicy()

    result = policy.validate("pkill -f python")

    assert not result.allowed
    assert "allowlist" in result.reason


def test_block_working_directory_outside_sandbox():
    policy = CommandSecurityPolicy()

    result = policy.validate("ls", working_directory="/tmp")

    assert not result.allowed
    assert "outside the sandbox" in result.reason
