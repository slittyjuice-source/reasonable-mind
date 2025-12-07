# Change Log

## 2025-05-25
- Refined `CODE_REVIEW.md` with clearer scope, detailed findings, and actionable next steps for the agent loop, history truncation, and tool error handling.

## 2025-05-26
- Added severity and success metrics to `CODE_REVIEW.md` findings to make priorities and expected outcomes explicit.
- Updated `MessageHistory.truncate` to remove messages one at a time based on tracked token usage and insert a single truncation notice while respecting the context window.
