# SmolDataEnvs task image + Hermes Agent v2026.9.24 preinstalled (same tag-matched installer the patched
# Harbor adapter runs per trial). Only removes the per-trial ~5 min install; the agent itself is unchanged.
FROM docker.io/savatar101/env-data-agent-train:base
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git ripgrep xz-utils build-essential libatomic1 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/v2026.9.24/scripts/install.sh | bash -s -- --skip-setup --branch v2026.9.24
RUN bash -c 'export PATH="$HOME/.local/bin:$PATH"; hermes --version'
WORKDIR /workdir
