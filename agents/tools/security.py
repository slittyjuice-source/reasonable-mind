"""Shared security utilities mirroring the autonomous-coding sandbox."""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field
from typing import Iterable, List, Set


DEFAULT_ALLOWED_COMMANDS: Set[str] = {
    # File inspection
    "ls",
    "cat",
    "head",
    "tail",
    "wc",
    "grep",
    # File operations
    "cp",
    "mkdir",
    "chmod",
    # Directory
    "pwd",
    # Node.js development
    "npm",
    "node",
    # Version control
    "git",
    # Process management
    "ps",
    "lsof",
    "sleep",
    "pkill",
    # Script execution
    "init.sh",
}

COMMANDS_NEEDING_EXTRA_VALIDATION = {"pkill", "chmod", "init.sh"}


@dataclass
class ValidationResult:
    allowed: bool
    reason: str = ""


def split_command_segments(command_string: str) -> list[str]:
    """Split compound commands on &&, ||, and ; while respecting quotes."""

    segments = re.split(r"\s*(?:&&|\|\|)\s*", command_string)
    result: list[str] = []
    for segment in segments:
        sub_segments = re.split(r"(?<![\"'])\s*;\s*(?![\"'])", segment)
        for sub in sub_segments:
            sub = sub.strip()
            if sub:
                result.append(sub)
    return result


def extract_commands(command_string: str) -> list[str]:
    """Extract command names from a shell string, handling pipes and chaining."""

    commands: list[str] = []
    segments = re.split(r"(?<![\"'])\s*;\s*(?![\"'])", command_string)

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        try:
            tokens = shlex.split(segment)
        except ValueError:
            return []

        if not tokens:
            continue

        expect_command = True
        for token in tokens:
            if token in ("|", "||", "&&", "&"):
                expect_command = True
                continue
            if token in (
                "if",
                "then",
                "else",
                "elif",
                "fi",
                "for",
                "while",
                "until",
                "do",
                "done",
                "case",
                "esac",
                "in",
                "!",
                "{",
                "}",
            ):
                continue
            if token.startswith("-"):
                continue
            if "=" in token and not token.startswith("="):
                continue
            if expect_command:
                commands.append(os.path.basename(token))
                expect_command = False

    return commands


def validate_pkill_command(command_string: str) -> ValidationResult:
    """Only allow pkill for common dev processes."""

    allowed_process_names = {"node", "npm", "npx", "vite", "next"}

    try:
        tokens = shlex.split(command_string)
    except ValueError:
        return ValidationResult(False, "Could not parse pkill command")

    if len(tokens) < 2:
        return ValidationResult(False, "pkill requires a process name")

    args = [token for token in tokens[1:] if not token.startswith("-")]
    if not args:
        return ValidationResult(False, "pkill requires a process name")

    target = args[-1]
    process = target.split()[0]

    if process not in allowed_process_names:
        return ValidationResult(False, f"pkill target '{process}' is not in allowlist")

    return ValidationResult(True)


def validate_chmod_command(command_string: str) -> ValidationResult:
    """Allow chmod only for making scripts executable inside allowed roots."""

    try:
        tokens = shlex.split(command_string)
    except ValueError:
        return ValidationResult(False, "Could not parse chmod command")

    if len(tokens) < 3:
        return ValidationResult(False, "chmod requires mode and target")

    mode = tokens[1]
    if mode not in {"+x", "755", "0755"}:
        return ValidationResult(False, "chmod may only set executable bits")

    return ValidationResult(True)


def validate_init_script(command_string: str) -> ValidationResult:
    """Block running arbitrary init scripts; require explicit allowlisting."""

    return ValidationResult(False, "init scripts must be reviewed before execution")


def validate_commands(segment: str, commands: Iterable[str]) -> ValidationResult:
    for command in commands:
        if command == "pkill":
            return validate_pkill_command(segment)
        if command == "chmod":
            return validate_chmod_command(segment)
        if command == "init.sh":
            return validate_init_script(segment)
    return ValidationResult(True)


@dataclass
class CommandSecurityPolicy:
    """Allowlist-based sandbox policy for bash execution."""

    allowed_commands: Set[str] = field(
        default_factory=lambda: set(DEFAULT_ALLOWED_COMMANDS)
    )
    allowed_paths: List[str] = field(
        default_factory=lambda: [os.path.abspath(os.getcwd())]
    )

    def is_path_allowed(self, directory: str | None) -> bool:
        if directory is None:
            return True

        abs_dir = os.path.abspath(directory)
        return any(
            abs_dir.startswith(os.path.abspath(path)) for path in self.allowed_paths
        )

    def validate(
        self, command: str, working_directory: str | None = None
    ) -> ValidationResult:
        if not command.strip():
            return ValidationResult(False, "Command is empty")

        segments = split_command_segments(command)
        for segment in segments:
            commands = extract_commands(segment)
            if not commands:
                return ValidationResult(False, "Unable to parse command")

            for cmd in commands:
                if cmd not in self.allowed_commands:
                    return ValidationResult(False, f"Command '{cmd}' is not allowed")

            if any(cmd in COMMANDS_NEEDING_EXTRA_VALIDATION for cmd in commands):
                validation = validate_commands(segment, commands)
                if not validation.allowed:
                    return validation

        if not self.is_path_allowed(working_directory):
            return ValidationResult(False, "Working directory is outside the sandbox")

        return ValidationResult(True)
