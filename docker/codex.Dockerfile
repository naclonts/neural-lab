FROM node:22-bookworm-slim
ARG CODEX_VERSION=0.165.0
RUN npm install --global @openai/codex@${CODEX_VERSION} \
    && npm cache clean --force \
    && mkdir -p /request /answer /codex
ENV CODEX_HOME=/codex HOME=/tmp
WORKDIR /request
USER node
ENTRYPOINT ["codex"]
