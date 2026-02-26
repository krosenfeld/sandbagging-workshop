FROM python:3.11-slim

# Install build tools and Node.js (required for C extensions and the claude CLI)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates g++ && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install the Claude Code CLI globally
RUN npm install -g @anthropic-ai/claude-code

WORKDIR /app

# Copy uv and project source
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/
COPY README.md pyproject.toml ./
COPY src/ src/

# Install all project dependencies
RUN uv pip install -e . --system

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Create non-root user (claude CLI refuses bypassPermissions as root)
RUN useradd -m -s /bin/bash agent
RUN chown agent:agent /app -R
RUN mkdir -p /home/agent/agent_logs /home/agent/workspaces && \
    chown agent:agent /home/agent/agent_logs /home/agent/workspaces

USER agent

EXPOSE 9100

ENTRYPOINT ["docker-entrypoint.sh"]
