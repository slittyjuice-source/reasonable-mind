# Contributing to Reasonable Mind

Thank you for contributing to Reasonable Mind! This document outlines the development workflow and standards for this project.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/reasonable-mind.git
cd reasonable-mind

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest agents/tests/ -v
```

## Code Standards

### Python Style
- Follow PEP 8 guidelines
- Use type hints for all function parameters and returns
- Prefer dataclasses for structured data
- Maximum line length: 100 characters

### Imports
- Use isort with `--combine-as-imports`
- Group imports: standard library, third-party, local

### Testing
- Write tests for all new functionality
- Maintain >80% test coverage
- Run full test suite before submitting PRs

## CHANGELOG Requirements

**IMPORTANT**: All code modifications MUST include a corresponding CHANGELOG entry.

### When to Update CHANGELOG

Update the CHANGELOG.md in the affected subdirectory for:
- ✅ New features or functionality
- ✅ Bug fixes
- ✅ Security patches
- ✅ Breaking changes
- ✅ Deprecations
- ✅ Performance improvements
- ❌ Documentation-only changes (optional)
- ❌ Test-only changes (optional)

### CHANGELOG Location

Each subdirectory with code has its own CHANGELOG.md:
- `agents/core/CHANGELOG.md` - Core reasoning modules
- `agents/core_logic/CHANGELOG.md` - Logic engines
- `agents/governance/CHANGELOG.md` - Governance and security
- `agents/tools/CHANGELOG.md` - Agent tools
- `agents/logic/CHANGELOG.md` - Knowledge representation
- And so on...

### CHANGELOG Format

We follow [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format:

```markdown
## [Unreleased]

### Added
- **filename.py**: Brief description of new feature
  - Detailed explanation if needed
  - Additional context

### Changed
- **filename.py**: Description of modification
  - Impact on existing behavior

### Fixed
- **filename.py**: Description of bug fix
  - Root cause if relevant

### Security
- **filename.py**: Description of security patch
  - CVE or vulnerability details if applicable

### Deprecated
- **filename.py**: Feature marked for removal
  - Migration path

### Removed
- **filename.py**: Removed feature
  - Replacement or alternative

### Known Issues
- **filename.py**: Description of unresolved issue
  - Impact and workaround if available
  - TODO or tracking reference
```

### Example CHANGELOG Entry

```markdown
## [Unreleased]

### Added
- **execution_proxy.py**: Enhanced shell injection prevention
  - Added detection for chained rm/curl/wget commands
  - Added command substitution blocking ($(...) and backticks)
  - Added protection against writing to system directories

### Security
- **execution_proxy.py**: Strengthened protection against shell injection
  - Added 8 new denylist patterns
  - Prevents: `ls; rm -rf /`, `echo $(whoami)`, `cat file | sh`
```

## Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code following style guidelines
   - Add tests for new functionality
   - **Update CHANGELOG.md** in affected subdirectories

3. **Verify Changes**
   ```bash
   # Run tests
   pytest agents/tests/ -v

   # Check formatting
   ruff check .
   ruff format .

   # Type check
   pyright
   ```

4. **Commit with Conventional Commits**
   ```bash
   git commit -m "feat: add categorical reasoning for second-figure syllogisms"
   git commit -m "fix: resolve shell injection vulnerability in execution proxy"
   git commit -m "docs: update CHANGELOG for governance module"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   # Create pull request on GitHub
   ```

6. **PR Checklist**
   - [ ] Tests pass
   - [ ] CHANGELOG.md updated in affected subdirectories
   - [ ] Type hints added
   - [ ] Documentation updated if needed
   - [ ] No security vulnerabilities introduced
   - [ ] Breaking changes documented

## Code Review

All submissions require review. Reviewers will check:
- Code quality and style adherence
- Test coverage
- **CHANGELOG completeness and accuracy**
- Security considerations
- Performance implications

## Security

Report security vulnerabilities privately via email rather than opening public issues.

## License

By contributing, you agree that your contributions will be licensed under the project's license.

## Questions?

Open a discussion on GitHub or reach out to the maintainers.
