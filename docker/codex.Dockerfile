FROM node:22-bookworm-slim
ARG CODEX_VERSION=0.165.0
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-pip git ca-certificates curl ripgrep sudo util-linux build-essential procps \
    && rm -rf /var/lib/apt/lists/* \
    && npm install --global @openai/codex@${CODEX_VERSION} \
    && npm cache clean --force \
    && mkdir -p /request /answer /codex
ENV CODEX_HOME=/codex HOME=/tmp
COPY docker/research-entrypoint.sh /usr/local/bin/research-entrypoint
RUN chmod 755 /usr/local/bin/research-entrypoint
WORKDIR /request
USER node
ENTRYPOINT ["codex"]
