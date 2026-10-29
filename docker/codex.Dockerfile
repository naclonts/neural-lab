FROM node:22-bookworm-slim
ARG CODEX_VERSION=0.165.0
RUN apt-get update && apt-get install -y --no-install-recommends python3 git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && npm install --global @openai/codex@${CODEX_VERSION} \
    && npm cache clean --force \
    && mkdir -p /request /answer /codex
ENV CODEX_HOME=/codex HOME=/tmp
WORKDIR /request
USER node
ENTRYPOINT ["codex"]
