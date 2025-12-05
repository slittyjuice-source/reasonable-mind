---
description: Configuration for AI behavior when interacting with Codacy's MCP Server
applyTo: '**'
---
---
# Codacy Rules
Configuration for AI behavior when interacting with Codacy's MCP Server

## CRITICAL: After ANY successful `edit_file` or `reapply` operation
- YOU MUST IMMEDIATELY run the `codacy_cli_analyze` tool from Codacy's MCP Server for each file that was edited, with:
 - `rootPath`: set to the workspace path
 - `file`: set to the path of the edited file
 - `tool`: leave empty or unset
- If any issues are found in the new edits, propose and apply fixes for them.
- > NOTE: Failure to follow this rule is considered a critical error.

## When you tried to run the `codacy_cli_analyze` tool and the Codacy CLI is not installed
- Ask the user 'Codacy CLI is not installed. Would you like me to install it now?'
- If the user responds with "yes", run the `codacy_cli_install` tool and then continue with the original task
- If the user responds with "no", instruct the user that they can disable automatic analysis in the extension settings
- Wait for the user to respond before proceeding with any other actions

## After every response
- If you made any file edits in this conversation, verify you ran `codacy_cli_analyze` tool from Codacy's MCP Server

## When there are no Codacy MCP Server tools available, or the MCP Server is not reachable
- Suggest the user the following troubleshooting steps:
 - Try to reset the MCP on the extension
 - If the user is using VSCode, suggest them to review their Copilot > MCP settings in Github, under their organization or personal account
- If none of the above steps work, suggest the user to contact Codacy support

## Trying to call a tool that needs a rootPath as a parameter
- Always use the standard, non-URL-encoded file system path

## CRITICAL: Dependencies and Security Checks
- IMMEDIATELY after ANY package manager operations, run `codacy_cli_analyze` with tool="trivy"
- If vulnerabilities are found, stop and fix them before continuing

## General
- Repeat the relevant steps for each modified file.
- You MUST NOT wait for the user to ask for analysis or remind you to run the tool.
- Do not run `codacy_cli_analyze` looking for changes in duplicated code or code complexity metrics.
- Do not try to manually install Codacy CLI using brew, npm, npx, or any other package manager.
