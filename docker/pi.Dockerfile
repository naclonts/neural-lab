FROM node:24-bookworm-slim
ARG PI_VERSION=1.5.0
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-pip git ca-certificates curl ripgrep sudo util-linux build-essential procps \
    && rm -rf /var/lib/apt/lists/* \
    && npm install --global --ignore-scripts @earendil-works/pi-coding-agent@${PI_VERSION} \
    && npm cache clean --force \
    && mkdir -p /request /answer /pi
ENV PI_CODING_AGENT_DIR=/pi HOME=/tmp
COPY docker/research-entrypoint.sh /usr/local/bin/research-entrypoint
RUN chmod 755 /usr/local/bin/research-entrypoint
WORKDIR /request
USER node
ENTRYPOINT ["pi"]
