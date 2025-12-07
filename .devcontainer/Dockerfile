# syntax=docker/dockerfile:1

# Use a standard devcontainers base image
FROM mcr.microsoft.com/devcontainers/base:bookworm

# CRITICAL: ensure all later RUN + feature scripts use bash, not sh
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Basic tooling (often already there, but safe)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Default working directory for VS Code
WORKDIR /workspaces/reasonable-mind